#!/usr/bin/env python3
"""
🦦 Ensemble V3 — 三模型加权融合（支持邻域增强V3）

更新: 
- 集成 NeighborAwareTransformer 推理
- 邻域先验在V3分支推理时启用
- Ensemble融合时加入邻域先验矫正
"""
import csv, json, os, sys, math, time, subprocess, re
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb

# ===== 配置 =====
DATA_PATH = "./data/lottery_history.csv"
V3_MODEL_DIR = "./models/v3"
XGB_MODEL_DIR = "./models/xgb"
LLM_MODEL_PATH = os.path.expanduser("~/qwen2.5-1.5b-q4.gguf")
LLAMA_SERVER_PATH = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
PRED_DIR = "./predictions"
os.makedirs(PRED_DIR, exist_ok=True)

SEQ_LEN = 8
VEC_DIM = 343
NEI_DIM = 49
TOTAL_DIM = VEC_DIM + NEI_DIM + NEI_DIM
LLM_PORT = 8081

# ===== 邻域先验权重 =====
NEIGHBORHOOD_ALPHA = 0.25  # 邻域先验融合权重

# ===== 数据加载 =====
def load_data():
    rows = []
    with open(DATA_PATH) as f:
        reader = csv.reader(f)
        h = next(reader)
        for r in reader:
            nums = []
            for i in [2, 6, 10, 14, 18, 22]:
                if i < len(r) and r[i].strip():
                    try: nums.append(int(r[i]))
                    except: break
            if len(nums) != 6: continue
            spc = None
            if 26 < len(r) and r[26].strip():
                try: spc = int(r[26])
                except: pass
            rows.append((nums, spc))
    return rows

rows = load_data()
print(f"📊 数据: {len(rows)}期")
print(f"  上期: {rows[-1][0]} + T{rows[-1][1]}")

# ===== 编码 =====
def encode_draw(nums, spc=None):
    vec = np.zeros(VEC_DIM, dtype=np.float32)
    for i, n in enumerate(nums):
        if 1 <= n <= 49: vec[i * 49 + (n - 1)] = 1.0
    if spc and 1 <= spc <= 49: vec[6 * 49 + (spc - 1)] = 1.0
    return vec

def encode_neighborhood_mask(prev_nums, prev_spc):
    nei = np.zeros(49, dtype=np.float32)
    all_prev = list(prev_nums)
    if prev_spc: all_prev.append(prev_spc)
    sigma = 2.5
    for n in range(1, 50):
        min_dist = min(abs(n - p) for p in all_prev)
        nei[n-1] = math.exp(-(min_dist**2) / (2 * sigma**2))
    return nei

def encode_neighborhood_distance(prev_nums, prev_spc):
    nei = np.zeros(49, dtype=np.float32)
    all_prev = list(prev_nums)
    if prev_spc: all_prev.append(prev_spc)
    for n in range(1, 50):
        min_dist = min(abs(n - p) for p in all_prev)
        nei[n-1] = min_dist / 24.0
    return nei

# ==========================================
# 🅰 V3 Neighbor Transformer
# ==========================================
class NeighborhoodEnhancedTransformer(nn.Module):
    def __init__(self, input_dim=TOTAL_DIM, d_model=512, nhead=8, num_layers=6, dim_feedforward=1024):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=0.2,
            activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_fc = nn.Sequential(
            nn.Linear(d_model * SEQ_LEN, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, d_model))
        self.output_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 49))
            for _ in range(7)])
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.input_proj(x) + self.pos_encoder
        x = self.transformer(x)
        global_feat = self.global_fc(x.reshape(B, -1))
        last = x[:, -1, :] + global_feat * 0.2
        last = self.dropout(last)
        logits = torch.stack([head(last) for head in self.output_heads], dim=1)
        return logits

# ===== V3推理（含邻域先验融合） =====
def predict_v3(rows, model_path="models/v3/best_model.pth"):
    print("\n🅰 V3 Neighbor Enhanced...")
    
    if not os.path.exists(model_path):
        print("  ⚠️ 模型不存在，跳过")
        return None
    
    device = torch.device("cpu")
    model = NeighborhoodEnhancedTransformer()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("  ✅ 模型加载成功")
    
    # 多窗口voting
    votes = np.zeros((7, 49))
    contexts = [3, 5, 8, 10]
    
    # 获取上一期数据（用于邻域编码）
    prev_nums, prev_spc = rows[-2]
    nei_mask = encode_neighborhood_mask(prev_nums, prev_spc)
    nei_dist = encode_neighborhood_distance(prev_nums, prev_spc)
    
    for ctx in contexts:
        if ctx >= len(rows): continue
        x_vecs = []
        for i in range(len(rows) - ctx, len(rows)):
            nums, spc = rows[i]
            x_vecs.append(encode_draw(nums, spc))
        x = np.array(x_vecs)
        if ctx < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - ctx, VEC_DIM))
            x = np.vstack([pad, x])
        elif ctx > SEQ_LEN:
            x = x[-SEQ_LEN:]
        
        nei_mask_stacked = np.tile(nei_mask, (SEQ_LEN, 1))
        nei_dist_stacked = np.tile(nei_dist, (SEQ_LEN, 1))
        combined = np.concatenate([x, nei_mask_stacked, nei_dist_stacked], axis=1)
        
        with torch.no_grad():
            logits = model(torch.FloatTensor(combined).unsqueeze(0))
            probs = F.softmax(logits, dim=-1).squeeze(0)
            # 邻域先验融合
            enhanced = (1 - NEIGHBORHOOD_ALPHA) * probs + NEIGHBORHOOD_ALPHA * torch.FloatTensor(nei_mask).unsqueeze(0)
            votes += enhanced.numpy()
    
    # 选号
    votes_flat = votes[:6].sum(axis=0)
    best = np.argsort(votes_flat)[-15:][::-1]
    nums = []
    for b in best:
        if b + 1 not in nums:
            nums.append(b + 1)
        if len(nums) == 6: break
    for n in range(1, 50):
        if n not in nums and len(nums) < 6:
            nums.append(n)
    spc = int(votes[6].argmax()) + 1
    
    result = {"nums": [int(x) for x in sorted(nums)], "spc": int(spc)}
    print(f"  🎯 {result['nums']} + T{result['spc']}")
    return result

# ==========================================
# 🅱 XGBoost
# ==========================================
def predict_xgb(rows):
    print("\n🅱 XGBoost...")
    
    xgb_result_path = f"{XGB_MODEL_DIR}/xgb_result.json"
    
    if os.path.exists(xgb_result_path):
        with open(xgb_result_path) as f:
            cached = json.load(f)
        print(f"  加载缓存: {xgb_result_path}")
        # 检查是否基于相同数据
        print(f"  🎯 {cached['nums']} + T{cached['spc']}")
        return cached
    
    # 重新运行预测
    if not all(os.path.exists(f"{XGB_MODEL_DIR}/xgb_pos{p+1}.json") for p in range(6)) or \
       not os.path.exists(f"{XGB_MODEL_DIR}/xgb_spc.json"):
        print("  ⚠️ XGB模型不存在")
        return None
    
    # 加载模型并预测
    from ft_xgb_lottery import extract_features
    import xgboost as xgb
    
    models_pos = []
    for pi in range(6):
        m = xgb.Booster()
        m.load_model(f"{XGB_MODEL_DIR}/xgb_pos{pi+1}.json")
        models_pos.append(m)
    model_spc = xgb.Booster()
    model_spc.load_model(f"{XGB_MODEL_DIR}/xgb_spc.json")
    
    # 构建特征
    from ft_xgb_lottery import load_data
    all_data = load_data()
    all_X, _ = extract_features(all_data)
    latest_X = all_X[-1:].reshape(-1)
    
    # 预测每个位置
    nums = []
    for pi in range(6):
        x_input = np.concatenate([latest_X, [n for n in nums]] if nums else [latest_X])
        if len(nums) == 0:
            x_input = latest_X
        else:
            x_input = np.concatenate([latest_X, [n/49.0 for n in nums] + [0]*(6-len(nums))])
        pred = models_pos[pi].predict(xgb.DMatrix(x_input.reshape(1, -1)))
        nums.append(int(pred[0]) + 1)
    
    spc = int(model_spc.predict(xgb.DMatrix(latest_X.reshape(1, -1)))[0]) + 1
    
    result = {"nums": nums, "spc": spc}
    print(f"  🎯 {result['nums']} + T{result['spc']}")
    return result

# ==========================================
# 🅲 LLM (Qwen2.5-1.5B)
# ==========================================
def predict_llm(rows, num_runs=3):
    print("\n🅲 LLM (Qwen2.5-1.5B)...")
    
    if not os.path.exists(LLM_MODEL_PATH):
        print("  ⚠️ LLM模型不存在，跳过")
        return None
    
    try:
        sys.path.insert(0, "./models/llm")
        from infer import start_server, stop_server, predict_ensemble
    except:
        print("  ⚠️ LLM推理脚本不可用，跳过")
        return None
    
    if not start_server():
        print("  ⚠️ llama-server 启动失败")
        return None
    
    try:
        result = predict_ensemble(num_runs=num_runs)
        print(f"  🎯 {result[0]} + T{result[1]}")
        return {"nums": result[0], "spc": result[1]}
    except Exception as e:
        print(f"  ⚠️ LLM推理失败: {e}")
        return None
    finally:
        stop_server()

# ==========================================
# 🔗 Ensemble融合
# ==========================================
def ensemble(results, weights=None):
    """
    加权融合多模型结果
    策略: 每个模型给每个数字+权重投票，取Top6+Top1
    """
    if weights is None:
        weights = {"v3": 0.40, "xgb": 0.30, "llm": 0.30}
    
    votes = np.zeros(49)
    spc_votes = np.zeros(49)
    
    model_keys = {"v3": 0, "xgb": 1, "llm": 2}
    model_names = ["v3", "xgb", "llm"]
    
    print(f"\n🔗 Ensemble加权融合: {weights}")
    
    active_models = [k for k in model_names if results.get(k) is not None]
    total_weight = sum(weights[k] for k in active_models)
    
    for name in active_models:
        w = weights[name] / total_weight  # 归一化
        r = results[name]
        for n in r["nums"]:
            if 1 <= n <= 49:
                votes[n-1] += w
        if r["spc"] and 1 <= r["spc"] <= 49:
            spc_votes[r["spc"]-1] += w
        
        print(f"  {name}: w={w:.2f}  {r['nums']} + T{r['spc']}")
    
    # ===== 邻域后处理加成 =====
    prev_nums, prev_spc = rows[-1]
    for n in range(1, 50):
        min_dist = min(abs(n - p) for p in prev_nums)
        # 距离短就加分
        if min_dist <= 3:
            bonus = 0.15 * (1 - min_dist/4)  # 距离0=0.15, 距离3=0.0375
            votes[n-1] += bonus
            spc_votes[n-1] += bonus
    
    # 选号
    best = np.argsort(votes)[-15:][::-1]
    final_nums = []
    for b in best:
        if b + 1 not in final_nums:
            final_nums.append(b + 1)
        if len(final_nums) == 6: break
    for n in range(1, 50):
        if n not in final_nums and len(final_nums) < 6:
            final_nums.append(n)
    
    final_spc = int(spc_votes.argmax()) + 1
    
    return {
        "nums": sorted(final_nums),
        "spc": final_spc,
        "raw_votes": votes.tolist(),
        "spc_votes": spc_votes.tolist(),
    }

# ==========================================
# 🚀 主流程
# ==========================================
print("\n" + "=" * 60)
print("🦦 獭獭彩票预测引擎 V3 (Neighborhood Enhanced)")
print("=" * 60)

# 运行各模型
results = {}

# V3
results["v3"] = predict_v3(rows)

# XGBoost
results["xgb"] = predict_xgb(rows)

# LLM
results["llm"] = predict_llm(rows, num_runs=3)

# Ensemble
ensemble_weights = {"v3": 0.35, "xgb": 0.35, "llm": 0.30}
final = ensemble(results, weights=ensemble_weights)

print("\n" + "=" * 60)
print(f"🎯 最终预测: {final['nums']} + T{final['spc']}")
print(f"   全7: {final['nums'] + [final['spc']]}")
print("=" * 60)

# 保存
date_str = "2026-06-25"
prediction = {
    "engine": "Ensemble_V3_Neighborhood",
    "date": date_str,
    "params": {
        "neighborhood_alpha": NEIGHBORHOOD_ALPHA,
        "ensemble_weights": ensemble_weights,
        "voting_contexts": [3, 5, 8, 10],
    },
    "ensemble": {
        "nums": [int(x) for x in final["nums"]],
        "spc": int(final["spc"]),
    },
    "individual_models": {
        k: {"nums": [int(x) for x in v["nums"]], "spc": int(v["spc"])}
        for k, v in results.items() if v is not None
    },
}

pred_path = f"{PRED_DIR}/prediction_{date_str}.json"
with open(pred_path, "w") as f:
    json.dump(prediction, f, indent=2)

# 也写一份 latest
with open(f"{PRED_DIR}/prediction_latest.json", "w") as f:
    json.dump(prediction, f, indent=2)

print(f"\n✅ 已保存: {pred_path}")
