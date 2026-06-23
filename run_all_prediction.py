#!/usr/bin/env python3
"""
🦦 獭獭彩票预测引擎 — 一键跑全部模型
功能:
  1. 加载最新数据
  2. 跑 V3 Transformer 推理
  3. 跑 XGBoost 推理
  4. 跑 Ensemble (加权融合)
  5. 给出综合推荐
  6. 保存结果到 daily_predictions/
"""
import csv, json, os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from datetime import datetime
from collections import Counter

# ===== 配置 =====
DATA_PATH = "/home/ubuntu/lottery_bert_research/data/lottery_all_years_updated_20260423.csv"
V3_MODEL_DIR = "/home/ubuntu/lottery_bert_research/ft_model_v3"
XGB_MODEL_DIR = "/home/ubuntu/lottery_bert_research/ft_xgb_model"
PRED_DIR = "/home/ubuntu/lottery_bert_research/daily_predictions"
os.makedirs(PRED_DIR, exist_ok=True)

SEQ_LEN = 8
VEC_DIM = 343

# ===== 1. 加载数据 =====
print("📊 加载数据...")
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

print(f"  总期数: {len(rows)}")
last_period = rows[-1]
print(f"  上期: {last_period[0]} + T{last_period[1]}")

# ===== 2. 工具函数 =====
def encode_draw(nums, spc=None):
    vec = np.zeros(VEC_DIM, dtype=np.float32)
    for i, n in enumerate(nums):
        if 1 <= n <= 49: vec[i * 49 + (n - 1)] = 1.0
    if spc and 1 <= spc <= 49: vec[6 * 49 + (spc - 1)] = 1.0
    return vec

# ===== 3. V3 Transformer 模型 =====
print("\n🅰 加载 V3 Transformer...")

class LotteryTransformerV3(nn.Module):
    def __init__(self, input_dim=343, d_model=512, nhead=8, num_layers=6, dim_feedforward=1024):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, dropout=0.2, activation='gelu', batch_first=True)
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
        outputs = [head(last) for head in self.output_heads]
        return torch.stack(outputs, dim=1)

device = torch.device("cpu")

v3_model = None
v3_loaded = False
v3_ok = os.path.exists(f"{V3_MODEL_DIR}/best_model.pth")
if v3_ok:
    try:
        v3_model = LotteryTransformerV3()
        v3_model.load_state_dict(torch.load(f"{V3_MODEL_DIR}/best_model.pth", map_location=device, weights_only=True))
        v3_model.eval()
        v3_loaded = True
        print("  ✅ V3 加载成功!")
    except Exception as e:
        print(f"  ❌ V3 加载失败: {e}")
else:
    print("  ⚠️ V3 模型文件不存在, 跳过")

# ===== 4. XGBoost 模型 =====
print("\n🅱 加载 XGBoost...")

xgb_loaded = False
models_pos = {}
model_spc = None
feat_names = None

if all(os.path.exists(f"{XGB_MODEL_DIR}/xgb_pos{pi+1}.json") for pi in range(6)) and \
   os.path.exists(f"{XGB_MODEL_DIR}/xgb_spc.json"):
    try:
        # 从保存的配置恢复特征名
        result_path = f"{XGB_MODEL_DIR}/xgb_result.json"
        if os.path.exists(result_path):
            with open(result_path) as f:
                result_data = json.load(f)
            feat_names = result_data['train_config']['feature_names']
            print(f"  特征维度: {len(feat_names)}")
        
        for pi in range(6):
            model = xgb.Booster()
            model.load_model(f"{XGB_MODEL_DIR}/xgb_pos{pi+1}.json")
            models_pos[pi] = model
        
        model_spc = xgb.Booster()
        model_spc.load_model(f"{XGB_MODEL_DIR}/xgb_spc.json")
        xgb_loaded = True
        print("  ✅ XGBoost 加载成功!")
    except Exception as e:
        print(f"  ❌ XGBoost 加载失败: {e}")
else:
    print("  ⚠️ XGBoost 模型文件不存在, 跳过")

# ===== 5. 运行 V3 预测 =====
v3_votes = None
v3_pred = None

if v3_loaded:
    print("\n🔮 V3 预测中...")
    votes = np.zeros((7, 49))
    window_sizes = [3, 5, 8, 10]
    
    for ctx in window_sizes:
        if ctx > len(rows): continue
        seq_nums = [rows[-i][0] for i in range(ctx, 0, -1)]
        seq_spc = [rows[-i][1] for i in range(ctx, 0, -1)]
        x_vecs = np.array([encode_draw(n, s) for n, s in zip(seq_nums, seq_spc)])
        if ctx < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - ctx, VEC_DIM))
            x_vecs = np.vstack([pad, x_vecs])
        elif ctx > SEQ_LEN:
            x_vecs = x_vecs[-SEQ_LEN:]
        with torch.no_grad():
            pred = v3_model(torch.FloatTensor(x_vecs).unsqueeze(0))
        votes += F.softmax(pred, dim=-1).squeeze(0).numpy()
    
    v3_votes = votes
    v3_pred = votes.argmax(axis=1)
    print(f"  平码: {[int(v3_pred[i])+1 for i in range(6)]}")
    print(f"  特码: {int(v3_pred[6])+1}")
else:
    print("  ⏭️ 跳过 V3")

# ===== 6. 运行 XGBoost 预测 =====
xgb_votes = None
xgb_pred = None

if xgb_loaded and feat_names:
    print("\n🔮 XGBoost 预测中...")
    
    def extract_features(rows, idx):
        """从第idx期之前提取特征（与训练时一致）"""
        if idx < 1: return None
        past = rows[:idx]
        prev = rows[idx-1] if idx >= 1 else None
        
        feats = {}
        
        # 各位置历史频率
        pos_freq = np.zeros((6, 49), dtype=np.float32)
        spc_freq = np.zeros(49, dtype=np.float32)
        for pnums, pspc in past:
            for pi, n in enumerate(pnums):
                if 1 <= n <= 49: pos_freq[pi, n-1] += 1
            if pspc and 1 <= pspc <= 49: spc_freq[pspc-1] += 1
        total_past = len(past)
        for pi in range(6):
            feats[f'pos{pi+1}_freq'] = pos_freq[pi] / max(total_past, 1)
        feats['spc_freq'] = spc_freq / max(total_past, 1)
        
        # 全局频率
        global_freq = np.zeros(49, dtype=np.float32)
        for pnums, pspc in past:
            for n in pnums:
                if 1 <= n <= 49: global_freq[n-1] += 1
            if pspc and 1 <= pspc <= 49: global_freq[pspc-1] += 1
        feats['global_freq'] = global_freq / max(total_past, 1)
        
        # 近期频率
        for window, label in [(10, '10'), (20, '20'), (30, '30'), (50, '50')]:
            recent = past[-min(window, len(past)):]
            rfreq = np.zeros(49, dtype=np.float32)
            for pnums, pspc in recent:
                for n in pnums:
                    if 1 <= n <= 49: rfreq[n-1] += 1
                if pspc and 1 <= pspc <= 49: rfreq[pspc-1] += 1
            feats[f'recent_{label}_freq'] = rfreq / max(len(recent) * 7, 1)
        
        # Gap
        gaps = np.full(49, len(past), dtype=np.int32)
        for offset in range(len(past)):
            pnums, pspc = past[len(past)-1-offset]
            for n in list(pnums) + ([pspc] if pspc else []):
                if 1 <= n <= 49 and gaps[n-1] == len(past):
                    gaps[n-1] = offset
        feats['gap'] = gaps.astype(np.float32) / max(len(past), 1)
        
        gap_cat = np.zeros(49, dtype=np.int32)
        for n in range(49):
            if gaps[n] <= 3: gap_cat[n] = 0
            elif gaps[n] <= 10: gap_cat[n] = 1
            elif gaps[n] <= 30: gap_cat[n] = 2
            else: gap_cat[n] = 3
        feats['gap_cat'] = gap_cat.astype(np.float32)
        
        if prev:
            for pi, n in enumerate(prev[0]):
                feats[f'prev_pos{pi+1}'] = float(n) / 49.0
            feats['prev_spc'] = float(prev[1]) / 49.0 if prev[1] else 0.0
        
        if len(past) >= 2:
            last2 = past[-2:]
            for pi in range(6):
                diff = last2[1][0][pi] - last2[0][0][pi]
                feats[f'delta_pos{pi+1}'] = diff / 49.0
            if last2[1][1] and last2[0][1]:
                feats['delta_spc'] = (last2[1][1] - last2[0][1]) / 49.0
        
        odd_ratio = [sum(1 for n in pnums if n%2==1)/6.0 for pnums,_ in past[-30:]]
        feats['odds_ratio_mean'] = np.mean(odd_ratio) if odd_ratio else 0.5
        feats['odds_ratio_std'] = np.std(odd_ratio) if odd_ratio else 0.0
        
        sums = [sum(pnums) for pnums,_ in past[-30:]]
        feats['sum_mean'] = np.mean(sums) / 300.0 if sums else 0.5
        feats['sum_std'] = np.std(sums) / 300.0 if sums else 0.0
        
        spans = [(max(pnums)-min(pnums))/49.0 for pnums,_ in past[-30:]]
        feats['span_mean'] = np.mean(spans) if spans else 0.5
        feats['span_std'] = np.std(spans) if spans else 0.0
        
        recent_nums = set()
        for pnums, pspc in past[-20:]:
            for n in pnums: recent_nums.add(n)
            if pspc: recent_nums.add(pspc)
        feats['recent_unique'] = len(recent_nums) / 49.0
        
        return feats
    
    def flatten_feats(feats_dict):
        vec = []
        for key, val in feats_dict.items():
            if isinstance(val, np.ndarray):
                vec.extend(val.flatten())
            elif isinstance(val, (int, float, np.integer, np.floating)):
                vec.append(float(val))
        return np.array(vec, dtype=np.float32)
    
    # 多窗口预测
    window_sizes = [3, 5, 8, 10, 15]
    xgb_votes = np.zeros((7, 49))
    MIN_HISTORY = 60
    
    for ws in window_sizes:
        for step in range(ws):
            ref_idx = len(rows) - ws + step
            if ref_idx < MIN_HISTORY: continue
            feats = extract_features(rows, ref_idx)
            if feats is None: continue
            x = flatten_feats(feats).reshape(1, -1)
            dtest = xgb.DMatrix(x, feature_names=feat_names)
            for pi in range(6):
                probs = models_pos[pi].predict(dtest)[0]
                xgb_votes[pi] += probs
            spc_probs = model_spc.predict(dtest)[0]
            xgb_votes[6] += spc_probs
    
    xgb_votes /= (len(window_sizes) * len(window_sizes))
    xgb_pred = xgb_votes.argmax(axis=1)
    print(f"  平码: {[int(xgb_pred[i])+1 for i in range(6)]}")
    print(f"  特码: {int(xgb_pred[6])+1}")
else:
    print("  ⏭️ 跳过 XGBoost")

# ===== 7. Ensemble 融合 =====
print("\n" + "=" * 65)
print("🦦 獭獭综合推荐 — Ensemble 融合")
print("=" * 65)

ensemble_votes = None
ensemble_pred = None
weights = {}

if v3_votes is not None and xgb_votes is not None:
    # 加权融合：V3权重0.5, XGB权重0.5
    w_v3 = 0.5
    w_xgb = 0.5
    ensemble_votes = w_v3 * v3_votes + w_xgb * xgb_votes
    ensemble_pred = ensemble_votes.argmax(axis=1)
    weights = {"v3": w_v3, "xgb": w_xgb}
    print(f"\n  融合权重: V3={w_v3}, XGB={w_xgb}")
elif v3_votes is not None:
    ensemble_votes = v3_votes
    ensemble_pred = v3_pred
    weights = {"v3": 1.0}
    print(f"\n  仅 V3 可用")
elif xgb_votes is not None:
    ensemble_votes = xgb_votes
    ensemble_pred = xgb_pred
    weights = {"xgb": 1.0}
    print(f"\n  仅 XGB 可用")
else:
    print("  ❌ 无可用模型!")
    sys.exit(1)

# 综合推荐
final_nums = [int(ensemble_pred[i])+1 for i in range(6)]
final_spc = int(ensemble_pred[6])+1

print(f"\n🎯 综合推荐:")
print(f"  平码: {final_nums}")
print(f"  特码: {final_spc}")

# 各模型推荐对比
print(f"\n📊 模型对比:")
if v3_pred is not None:
    v3_nums = [int(v3_pred[i])+1 for i in range(6)]
    v3_spc_num = int(v3_pred[6])+1
    print(f"  🅰 V3:       {v3_nums} + T{v3_spc_num}")
if xgb_pred is not None:
    xgb_nums = [int(xgb_pred[i])+1 for i in range(6)]
    xgb_spc_num = int(xgb_pred[6])+1
    print(f"  🅱 XGBoost:  {xgb_nums} + T{xgb_spc_num}")
print(f"  🦦 Ensemble: {final_nums} + T{final_spc}")

# 各位置Top5 (融合后)
print(f"\n📈 每位置 Top5 (融合概率):")
for p in range(7):
    label = f"位置{p+1}" if p < 6 else "特码"
    top5 = np.argsort(ensemble_votes[p])[-5:][::-1]
    top5_str = ", ".join([f"{int(idx)+1}({ensemble_votes[p][idx]:.3f})" for idx in top5])
    print(f"  {label}: {top5_str}")

# 共识分析：看看V3和XGB在哪些号码上一致
if v3_pred is not None and xgb_pred is not None:
    agreed_positions = []
    for p in range(7):
        if v3_pred[p] == xgb_pred[p]:
            label = f"位置{p+1}" if p < 6 else "特码"
            num = int(v3_pred[p]) + 1
            agreed_positions.append(f"{label}={num}")
    
    if agreed_positions:
        print(f"\n🤝 双模型共识:")
        print(f"  {', '.join(agreed_positions)}")
    else:
        print(f"\n🤝 双模型无共识 (各推各的)")
    
    # 在top3范围内看重叠度
    overlap_counts = []
    for p in range(7):
        v3_top3 = set(np.argsort(v3_votes[p])[-3:])
        xgb_top3 = set(np.argsort(xgb_votes[p])[-3:])
        overlap = v3_top3 & xgb_top3
        overlap_counts.append(len(overlap))
    
    print(f"  Top3重叠: {overlap_counts[:6]} + 特码={overlap_counts[6]}")

# ===== 8. 保存结果 =====
date_str = "latest"
output = {
    "engine": "Ensemble_v1",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "data": {
        "total_periods": len(rows),
        "last_period": {"nums": last_period[0], "spc": last_period[1]}
    },
    "weights": weights,
    "v3_available": v3_loaded,
    "xgb_available": xgb_loaded,
    "ensemble": {
        "nums": final_nums,
        "spc": final_spc,
        "position_top5": {
            f"pos{p+1}" if p < 6 else "spc": [int(np.argsort(ensemble_votes[p])[-5:][::-1][i])+1 for i in range(5)]
            for p in range(7)
        }
    },
    "individual_models": {}
}

if v3_pred is not None:
    output["individual_models"]["v3"] = {
        "nums": [int(v3_pred[i])+1 for i in range(6)],
        "spc": int(v3_pred[6])+1
    }
if xgb_pred is not None:
    output["individual_models"]["xgb"] = {
        "nums": [int(xgb_pred[i])+1 for i in range(6)],
        "spc": int(xgb_pred[6])+1
    }

# 保存两份 (latest + 时间戳)
for suffix in ["latest", datetime.now().strftime("%Y%m%d_%H%M%S")]:
    path = f"{PRED_DIR}/prediction_{suffix}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 已保存: {path}")

print(f"\n⏱️ 完成! 🦦")
