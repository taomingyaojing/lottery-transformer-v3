#!/usr/bin/env python3
"""
🦦 獭獭彩票预测引擎 V2 — 三模型 Ensemble (V3 + XGBoost + LLM)
"""
import csv, json, os, sys, math, re, time, subprocess, urllib.request, urllib.error
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from collections import Counter
from datetime import datetime

# ===== 配置 =====
DATA_PATH = "/home/ubuntu/lottery_bert_research/data/lottery_all_years_updated_20260423.csv"
V3_MODEL_DIR = "/home/ubuntu/lottery_bert_research/ft_model_v3"
XGB_MODEL_DIR = "/home/ubuntu/lottery_bert_research/ft_xgb_model"
LLM_MODEL_PATH = os.path.expanduser("~/qwen2.5-1.5b-q4.gguf")
LLAMA_SERVER_PATH = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
PRED_DIR = "/home/ubuntu/lottery_bert_research/daily_predictions"
os.makedirs(PRED_DIR, exist_ok=True)

SEQ_LEN = 8
VEC_DIM = 343
LLM_PORT = 8081

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

# ===== V3 Transformer =====
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
v3_votes = None
v3_ok = os.path.exists(f"{V3_MODEL_DIR}/best_model.pth")
if v3_ok:
    try:
        v3_model = LotteryTransformerV3()
        v3_model.load_state_dict(torch.load(f"{V3_MODEL_DIR}/best_model.pth", map_location=device, weights_only=True))
        v3_model.eval()
        print("  ✅ V3 加载成功!")
        
        # V3预测
        votes = np.zeros((7, 49))
        for ctx in [3, 5, 8, 10]:
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
        print(f"  🎯 {[int(v3_pred[i])+1 for i in range(6)]} + T{int(v3_pred[6])+1}")
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
else:
    print("  ⏭️ 跳过")

# ===== XGBoost =====
print("\n🅱 加载 XGBoost...")
xgb_votes = None
models_pos = {}
model_spc = None
feat_names = None

xgb_ok = all(os.path.exists(f"{XGB_MODEL_DIR}/xgb_pos{pi+1}.json") for pi in range(6)) and \
         os.path.exists(f"{XGB_MODEL_DIR}/xgb_spc.json")

if xgb_ok:
    try:
        with open(f"{XGB_MODEL_DIR}/xgb_result.json") as f:
            feat_names = json.load(f)['train_config']['feature_names']
        for pi in range(6):
            m = xgb.Booster(); m.load_model(f"{XGB_MODEL_DIR}/xgb_pos{pi+1}.json"); models_pos[pi] = m
        model_spc = xgb.Booster(); model_spc.load_model(f"{XGB_MODEL_DIR}/xgb_spc.json")
        
        # XGBoost特征提取
        def extract_features(rows, idx):
            if idx < 1: return None
            past = rows[:idx]; prev = rows[idx-1] if idx >= 1 else None
            feats = {}
            pos_freq = np.zeros((6, 49), dtype=np.float32)
            spc_freq = np.zeros(49, dtype=np.float32)
            for pnums, pspc in past:
                for pi, n in enumerate(pnums):
                    if 1 <= n <= 49: pos_freq[pi, n-1] += 1
                if pspc and 1 <= pspc <= 49: spc_freq[pspc-1] += 1
            tp = len(past)
            for pi in range(6): feats[f'pos{pi+1}_freq'] = pos_freq[pi]/max(tp,1)
            feats['spc_freq'] = spc_freq/max(tp,1)
            gf = np.zeros(49)
            for pnums, pspc in past:
                for n in pnums:
                    if 1 <= n <= 49: gf[n-1] += 1
                if pspc and 1 <= pspc <= 49: gf[pspc-1] += 1
            feats['global_freq'] = gf/max(tp,1)
            for w, l in [(10,'10'),(20,'20'),(30,'30'),(50,'50')]:
                r = past[-min(w,len(past)):]
                rf = np.zeros(49)
                for pnums, pspc in r:
                    for n in pnums:
                        if 1 <= n <= 49: rf[n-1] += 1
                    if pspc and 1 <= pspc <= 49: rf[pspc-1] += 1
                feats[f'recent_{l}_freq'] = rf/max(len(r)*7,1)
            gaps = np.full(49, len(past), dtype=np.int32)
            for offset in range(len(past)):
                pnums, pspc = past[len(past)-1-offset]
                for n in list(pnums) + ([pspc] if pspc else []):
                    if 1 <= n <= 49 and gaps[n-1] == len(past): gaps[n-1] = offset
            feats['gap'] = gaps.astype(np.float32)/max(len(past),1)
            gc = np.zeros(49, dtype=np.int32)
            for n in range(49):
                if gaps[n] <= 3: gc[n]=0
                elif gaps[n] <= 10: gc[n]=1
                elif gaps[n] <= 30: gc[n]=2
                else: gc[n]=3
            feats['gap_cat'] = gc.astype(np.float32)
            if prev:
                for pi, n in enumerate(prev[0]): feats[f'prev_pos{pi+1}'] = float(n)/49.0
                feats['prev_spc'] = float(prev[1])/49.0 if prev[1] else 0.0
            if len(past) >= 2:
                l2 = past[-2:]
                for pi in range(6): feats[f'delta_pos{pi+1}'] = (l2[1][0][pi]-l2[0][0][pi])/49.0
                if l2[1][1] and l2[0][1]: feats['delta_spc'] = (l2[1][1]-l2[0][1])/49.0
            or_ = [sum(1 for n in pnums if n%2==1)/6.0 for pnums,_ in past[-30:]]
            feats['odds_ratio_mean'] = np.mean(or_) if or_ else 0.5
            feats['odds_ratio_std'] = np.std(or_) if or_ else 0.0
            ss = [sum(pnums) for pnums,_ in past[-30:]]
            feats['sum_mean'] = np.mean(ss)/300.0 if ss else 0.5
            feats['sum_std'] = np.std(ss)/300.0 if ss else 0.0
            sp = [(max(pnums)-min(pnums))/49.0 for pnums,_ in past[-30:]]
            feats['span_mean'] = np.mean(sp) if sp else 0.5
            feats['span_std'] = np.std(sp) if sp else 0.0
            rn = set()
            for pnums, pspc in past[-20:]:
                for n in pnums: rn.add(n)
                if pspc: rn.add(pspc)
            feats['recent_unique'] = len(rn)/49.0
            return feats
        
        def flatten_feats(fd):
            v = []
            for key, val in fd.items():
                if isinstance(val, np.ndarray): v.extend(val.flatten())
                elif isinstance(val, (int, float, np.integer, np.floating)): v.append(float(val))
            return np.array(v, dtype=np.float32)
        
        # XGBoost预测
        xgb_votes = np.zeros((7, 49))
        ws = [3, 5, 8, 10, 15]
        for w in ws:
            for step in range(w):
                ri = len(rows) - w + step
                if ri < 60: continue
                feats = extract_features(rows, ri)
                if feats is None: continue
                x = flatten_feats(feats).reshape(1, -1)
                d = xgb.DMatrix(x, feature_names=feat_names)
                for pi in range(6): xgb_votes[pi] += models_pos[pi].predict(d)[0]
                xgb_votes[6] += model_spc.predict(d)[0]
        xgb_votes /= (len(ws) * len(ws))
        xgb_pred = xgb_votes.argmax(axis=1)
        
        print("  ✅ XGBoost 加载成功!")
        print(f"  🎯 {[int(xgb_pred[i])+1 for i in range(6)]} + T{int(xgb_pred[6])+1}")
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
else:
    print("  ⏭️ 跳过")

# ===== LLM (Qwen2.5-1.5B) =====
print("\n🅲 启动 LLM (Qwen2.5-1.5B)...")
llm_nums = None
llm_spc = None
llm_ok = os.path.exists(LLM_MODEL_PATH) and os.path.exists(LLAMA_SERVER_PATH)

if llm_ok:
    try:
        # 启动server
        llm_proc = subprocess.Popen(
            [LLAMA_SERVER_PATH, "-m", LLM_MODEL_PATH, "--port", str(LLM_PORT),
             "--host", "127.0.0.1", "-c", "8192", "--no-mmap", "-ngl", "0"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        for i in range(30):
            try:
                with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{LLM_PORT}/health"), timeout=2):
                    break
            except:
                time.sleep(1)
        print("  ✅ LLM 加载成功!")
        
        # 多温度投票
        print("  🧠 多温度推理中...")
        llm_votes = Counter()
        
        for run, temp in enumerate([0.3, 0.45, 0.6, 0.75, 0.9]):
            recent = rows[-12:]
            seq_parts = ["-".join(f"{n:02d}" for n in ns) + f" {spc:02d}" for ns, spc in recent]
            prompt = f"Lottery sequence (last 12 draws):\n\n" + "\n".join(seq_parts) + "\n\nNext line:"
            
            data = json.dumps({"prompt": prompt, "n_predict": 30, "temperature": temp, "cache_prompt": False}).encode()
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(f"http://127.0.0.1:{LLM_PORT}/completion",
                        data=data, headers={"Content-Type": "application/json"}),
                    timeout=120
                ) as resp:
                    result = json.loads(resp.read()).get("content", "")
                    nums = [int(x) for x in re.findall(r'\b([1-9]|[1-4][0-9])\b', result)]
                    nums = [n for n in nums if 1 <= n <= 49]
                    print(f"    Run {run+1} (t={temp:.1f}): {nums[:8]}...")
                    for n in nums[:7]:
                        llm_votes[n] += 1
            except Exception as e:
                print(f"    Run {run+1}: ERROR {e}")
        
        # 取top6+1
        sorted_nums = [n for n, _ in llm_votes.most_common()]
        llm_nums = sorted_nums[:6]
        llm_spc = sorted_nums[6] if len(sorted_nums) > 6 else sorted_nums[0]
        
        print(f"  🎯 {llm_nums} + T{llm_spc}")
        
        llm_proc.terminate()
        try:
            llm_proc.wait(timeout=5)
        except:
            llm_proc.kill()
    except Exception as e:
        print(f"  ❌ LLM 失败: {e}")
else:
    print("  ⏭️ 跳过")

# ===== Ensemble =====
print("\n" + "=" * 65)
print("🦦 三模型 Ensemble 综合推荐")
print("=" * 65)

results = {}
votes_count = Counter()

if v3_votes is not None:
    v3_nums_final = [int(v3_pred[i])+1 for i in range(6)]
    v3_spc_final = int(v3_pred[6])+1
    results["V3"] = {"nums": v3_nums_final, "spc": v3_spc_final}
    for n in v3_nums_final + [v3_spc_final]:
        votes_count[n] += 1

if xgb_votes is not None:
    xgb_nums_final = [int(xgb_pred[i])+1 for i in range(6)]
    xgb_spc_final = int(xgb_pred[6])+1
    results["XGBoost"] = {"nums": xgb_nums_final, "spc": xgb_spc_final}
    for n in xgb_nums_final + [xgb_spc_final]:
        votes_count[n] += 1

if llm_nums is not None:
    results["LLM"] = {"nums": llm_nums, "spc": llm_spc}
    for n in llm_nums + [llm_spc]:
        votes_count[n] += 1

# Ensemble: 按票数排序
sorted_nums = [n for n, _ in votes_count.most_common()]
final_nums = []
final_spc = None
for n in sorted_nums:
    if len(final_nums) < 6:
        final_nums.append(n)
    elif final_spc is None:
        final_spc = n
        break

# 完整输出
print(f"\n📊 各模型推荐:")
for name, r in results.items():
    print(f"  {name}: {r['nums']} + T{r['spc']}")

print(f"\n🎯 综合推荐:")
print(f"  平码: {final_nums}")
print(f"  特码: {final_spc}")
print(f"  全7码: {'-'.join(f'{n:02d}' for n in final_nums)} + {final_spc:02d}")

# 共识分析
if len(results) >= 2:
    name_list = list(results.keys())
    for p in range(7):
        nums_set = set()
        for name in name_list:
            r = results[name]
            n = r['nums'][p] if p < 6 else r['spc']
            nums_set.add(n)
        if len(nums_set) == 1:
            label = f"位置{p+1}" if p < 6 else "特码"
            print(f"  🤝 {label}共识: {list(nums_set)[0]}")

# 保存
output = {
    "engine": "Ensemble_v2_3models",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "data": {"total_periods": len(rows), "last_period": {"nums": rows[-1][0], "spc": rows[-1][1]}},
    "models": results,
    "ensemble": {"nums": final_nums, "spc": final_spc}
}

for suffix in ["latest", datetime.now().strftime("%Y%m%d_%H%M%S")]:
    with open(f"{PRED_DIR}/prediction_{suffix}.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ 已保存: {PRED_DIR}/prediction_latest.json")
print(f"\n⏱️ 完成! 🦦")
