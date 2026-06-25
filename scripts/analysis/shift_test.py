#!/usr/bin/env python3
"""
🦓 变换实验：对数据做平移/放大，看预测是否同规律变化
"""
import csv, json, os
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH = "../data/lottery_history.csv"
V3_MODEL_DIR = "../models/v3"
XGB_MODEL_DIR = "../models/xgb"

SEQ_LEN = 8
VEC_DIM = 343

# ===== 原始数据加载 =====
def load_data(path):
    rows = []
    with open(path) as f:
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

rows = load_data(DATA_PATH)
print(f"原始数据: {len(rows)}期")
print(f"上期: {rows[-1]}")

# ===== 编码 =====
def encode_draw(nums, spc=None):
    vec = np.zeros(VEC_DIM, dtype=np.float32)
    for i, n in enumerate(nums):
        if 1 <= n <= 49: vec[i * 49 + (n - 1)] = 1.0
    if spc and 1 <= spc <= 49: vec[6 * 49 + (spc - 1)] = 1.0
    return vec

# ===== V3 模型 =====
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
v3_model = LotteryTransformerV3()
v3_model.load_state_dict(torch.load(f"{V3_MODEL_DIR}/best_model.pth", map_location=device, weights_only=True))
v3_model.eval()

# ===== XGBoost =====
result_path = f"{XGB_MODEL_DIR}/xgb_result.json"
with open(result_path) as f:
    result_data = json.load(f)
feat_names = result_data['train_config']['feature_names']

models_pos = {}
for pi in range(6):
    m = xgb.Booster()
    m.load_model(f"{XGB_MODEL_DIR}/xgb_pos{pi+1}.json")
    models_pos[pi] = m
model_spc = xgb.Booster()
model_spc.load_model(f"{XGB_MODEL_DIR}/xgb_spc.json")

# ===== XGB特征提取 =====
def extract_features(rows, idx):
    if idx < 1: return None
    past = rows[:idx]
    prev = rows[idx-1] if idx >= 1 else None
    feats = {}
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
    global_freq = np.zeros(49, dtype=np.float32)
    for pnums, pspc in past:
        for n in pnums:
            if 1 <= n <= 49: global_freq[n-1] += 1
        if pspc and 1 <= pspc <= 49: global_freq[pspc-1] += 1
    feats['global_freq'] = global_freq / max(total_past, 1)
    for window, label in [(10, '10'), (20, '20'), (30, '30'), (50, '50')]:
        recent = past[-min(window, len(past)):]
        rfreq = np.zeros(49, dtype=np.float32)
        for pnums, pspc in recent:
            for n in pnums:
                if 1 <= n <= 49: rfreq[n-1] += 1
            if pspc and 1 <= pspc <= 49: rfreq[pspc-1] += 1
        feats[f'recent_{label}_freq'] = rfreq / max(len(recent) * 7, 1)
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

# ===== 变换函数 =====
def shift_nums(nums, shift, mod=49):
    """平移变换"""
    return [((n - 1 + shift) % mod) + 1 for n in nums]

def scale_nums(nums, scale, mod=49):
    """等比例缩放（乘法）"""
    return [min(int(round(n * scale)), mod) for n in nums]

def invert_scale(n, scale, mod=49):
    """反向缩放"""
    if scale == 0: return 0
    return min(int(round(n / scale)), mod)

# ===== 预测函数 =====
def v3_predict(rows_data):
    votes = np.zeros((7, 49))
    for ctx in [3, 5, 8, 10]:
        if ctx > len(rows_data): continue
        seq_nums = [rows_data[-i][0] for i in range(ctx, 0, -1)]
        seq_spc = [rows_data[-i][1] for i in range(ctx, 0, -1)]
        x_vecs = np.array([encode_draw(n, s) for n, s in zip(seq_nums, seq_spc)])
        if ctx < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - ctx, VEC_DIM))
            x_vecs = np.vstack([pad, x_vecs])
        elif ctx > SEQ_LEN:
            x_vecs = x_vecs[-SEQ_LEN:]
        with torch.no_grad():
            pred = v3_model(torch.FloatTensor(x_vecs).unsqueeze(0))
        votes += F.softmax(pred, dim=-1).squeeze(0).numpy()
    return votes

def xgb_predict(rows_data, feat_names):
    votes = np.zeros((7, 49))
    window_sizes = [3, 5, 8, 10, 15]
    MIN_HISTORY = 60
    for ws in window_sizes:
        for step in range(ws):
            ref_idx = len(rows_data) - ws + step
            if ref_idx < MIN_HISTORY: continue
            feats = extract_features(rows_data, ref_idx)
            if feats is None: continue
            x = flatten_feats(feats).reshape(1, -1)
            dtest = xgb.DMatrix(x, feature_names=feat_names)
            for pi in range(6):
                probs = models_pos[pi].predict(dtest)[0]
                votes[pi] += probs
            spc_probs = model_spc.predict(dtest)[0]
            votes[6] += spc_probs
    votes /= (len(window_sizes) * len(window_sizes))
    return votes

# ===== 实验 =====
print("\n" + "=" * 80)
print("🦓 变换实验：预测是否随数据等比例变化？")
print("=" * 80)

# 实验1: 平移变换
print("\n" + "=" * 80)
print("📐 实验1: 平移变换 (全部号码 +5, 超出49从1开始)")
print("=" * 80)
SHIFT = 5

shifted_rows = [(shift_nums(ns, SHIFT), shift_nums([spc], SHIFT)[0] if spc else None) for ns, spc in rows]

print(f"\n原始上期:   {rows[-1][0]} + T{rows[-1][1]}")
print(f"平移后上期: {shifted_rows[-1][0]} + T{shifted_rows[-1][1]}")

# V3
v3_raw = v3_predict(rows)
v3_shifted = v3_predict(shifted_rows)
v3_raw_nums = [int(v3_raw.argmax(axis=1)[i])+1 for i in range(6)]
v3_raw_spc = int(v3_raw.argmax(axis=1)[6])+1
v3_shifted_nums = [int(v3_shifted.argmax(axis=1)[i])+1 for i in range(6)]
v3_shifted_spc = int(v3_shifted.argmax(axis=1)[6])+1
v3_expected = shift_nums(v3_raw_nums, SHIFT)

print(f"\n🅰 V3:")
print(f"  原始预测:       {v3_raw_nums} + T{v3_raw_spc}")
print(f"  平移后预测:     {v3_shifted_nums} + T{v3_shifted_spc}")
print(f"  理论上应平移后: {v3_expected} + T{shift_nums([v3_raw_spc], SHIFT)[0]}")
match_v3 = sum(1 for i in range(6) if v3_shifted_nums[i] == v3_expected[i])
spc_match = "✅" if v3_shifted_spc == shift_nums([v3_raw_spc], SHIFT)[0] else "❌"
print(f"  匹配度: 平码 {match_v3}/6, 特码 {spc_match}")

# XGBoost
xgb_raw = xgb_predict(rows, feat_names)
xgb_shifted = xgb_predict(shifted_rows, feat_names)
xgb_raw_nums = [int(xgb_raw.argmax(axis=1)[i])+1 for i in range(6)]
xgb_raw_spc = int(xgb_raw.argmax(axis=1)[6])+1
xgb_shifted_nums = [int(xgb_shifted.argmax(axis=1)[i])+1 for i in range(6)]
xgb_shifted_spc = int(xgb_shifted.argmax(axis=1)[6])+1
xgb_expected = shift_nums(xgb_raw_nums, SHIFT)

print(f"\n🅱 XGBoost:")
print(f"  原始预测:       {xgb_raw_nums} + T{xgb_raw_spc}")
print(f"  平移后预测:     {xgb_shifted_nums} + T{xgb_shifted_spc}")
print(f"  理论上应平移后: {xgb_expected} + T{shift_nums([xgb_raw_spc], SHIFT)[0]}")
match_xgb = sum(1 for i in range(6) if xgb_shifted_nums[i] == xgb_expected[i])
spc_match_xgb = "✅" if xgb_shifted_spc == shift_nums([xgb_raw_spc], SHIFT)[0] else "❌"
print(f"  匹配度: 平码 {match_xgb}/6, 特码 {spc_match_xgb}")

# 分析V3在平移前后Top5的变化
print(f"\n📊 V3 位置1 Top5变化:")
v3_raw_top5_1 = np.argsort(v3_raw[0])[-5:][::-1]
v3_shifted_top5_1 = np.argsort(v3_shifted[0])[-5:][::-1]
for i in range(5):
    print(f"  Top{i+1}: {v3_raw_top5_1[i]+1} → {v3_shifted_top5_1[i]+1} (期望: {(v3_raw_top5_1[i]+SHIFT-1)%49+1})")

# ===== 实验2: 等比例缩放 =====
print("\n" + "=" * 80)
print("📐 实验2: 等比例放大 (全部号码 ×0.7, 四舍五入)")
print("=" * 80)
SCALE = 0.7

scaled_rows = [(scale_nums(ns, SCALE), scale_nums([spc], SCALE)[0] if spc else None) for ns, spc in rows]

print(f"原始上期:   {rows[-1][0]} + T{rows[-1][1]}")
print(f"缩放后上期: {scaled_rows[-1][0]} + T{scaled_rows[-1][1]}")

# V3
v3_scaled = v3_predict(scaled_rows)
v3_scaled_nums = [int(v3_scaled.argmax(axis=1)[i])+1 for i in range(6)]
v3_scaled_spc = int(v3_scaled.argmax(axis=1)[6])+1
v3_expected_scaled = scale_nums(v3_raw_nums, SCALE)

print(f"\n🅰 V3:")
print(f"  原始预测:       {v3_raw_nums} + T{v3_raw_spc}")
print(f"  缩放后预测:     {v3_scaled_nums} + T{v3_scaled_spc}")
print(f"  理论上缩放后:   {v3_expected_scaled} + T{scale_nums([v3_raw_spc], SCALE)[0]}")
match_v3s = sum(1 for i in range(6) if v3_scaled_nums[i] == v3_expected_scaled[i])
print(f"  匹配度: 平码 {match_v3s}/6")

# XGBoost
xgb_scaled = xgb_predict(scaled_rows, feat_names)
xgb_scaled_nums = [int(xgb_scaled.argmax(axis=1)[i])+1 for i in range(6)]
xgb_scaled_spc = int(xgb_scaled.argmax(axis=1)[6])+1
xgb_expected_scaled = scale_nums(xgb_raw_nums, SCALE)

print(f"\n🅱 XGBoost:")
print(f"  原始预测:       {xgb_raw_nums} + T{xgb_raw_spc}")
print(f"  缩放后预测:     {xgb_scaled_nums} + T{xgb_scaled_spc}")
print(f"  理论上缩放后:   {xgb_expected_scaled} + T{scale_nums([xgb_raw_spc], SCALE)[0]}")
match_xgbs = sum(1 for i in range(6) if xgb_scaled_nums[i] == xgb_expected_scaled[i])
print(f"  匹配度: 平码 {match_xgbs}/6")

# ===== 实验3: 特码移位（仅改特码） =====
print("\n" + "=" * 80)
print("📐 实验3: 只改特码 (+10), 看平码预测是否变化")
print("=" * 80)

spc_shifted_rows = [(ns, ((spc - 1 + 10) % 49) + 1 if spc else None) for ns, spc in rows]
print(f"原始上期:   {rows[-1][0]} + T{rows[-1][1]}")
print(f"仅改特码后: {spc_shifted_rows[-1][0]} + T{spc_shifted_rows[-1][1]}")

# V3
v3_spc_shifted = v3_predict(spc_shifted_rows)
v3_spc_shifted_nums = [int(v3_spc_shifted.argmax(axis=1)[i])+1 for i in range(6)]
v3_spc_shifted_spc = int(v3_spc_shifted.argmax(axis=1)[6])+1
print(f"\n🅰 V3:")
print(f"  原始预测:         {v3_raw_nums} + T{v3_raw_spc}")
print(f"  改特码后预测:     {v3_spc_shifted_nums} + T{v3_spc_shifted_spc}")
match_v3ss = sum(1 for i in range(6) if v3_spc_shifted_nums[i] == v3_raw_nums[i])
print(f"  平码一致性: {match_v3ss}/6 (期望: 6/6 如果特码不影响平码)")

# XGBoost
xgb_spc_shifted = xgb_predict(spc_shifted_rows, feat_names)
xgb_spc_shifted_nums = [int(xgb_spc_shifted.argmax(axis=1)[i])+1 for i in range(6)]
xgb_spc_shifted_spc = int(xgb_spc_shifted.argmax(axis=1)[6])+1
print(f"\n🅱 XGBoost:")
print(f"  原始预测:         {xgb_raw_nums} + T{xgb_raw_spc}")
print(f"  改特码后预测:     {xgb_spc_shifted_nums} + T{xgb_spc_shifted_spc}")
match_xgbss = sum(1 for i in range(6) if xgb_spc_shifted_nums[i] == xgb_raw_nums[i])
print(f"  平码一致性: {match_xgbss}/6 (期望: 6/6 如果特码不影响平码)")

print("\n" + "=" * 80)
print("📋 结论分析")
print("=" * 80)
print(f"\n平移实验:")
print(f"  V3:      平码匹配 {match_v3}/6, 特码{'匹配' if v3_shifted_spc == shift_nums([v3_raw_spc], SHIFT)[0] else '不匹配'}")
print(f"  XGBoost: 平码匹配 {match_xgb}/6, 特码{'匹配' if xgb_shifted_spc == shift_nums([xgb_raw_spc], SHIFT)[0] else '不匹配'}")
print(f"\n缩放实验:")
print(f"  V3:      平码匹配 {match_v3s}/6")
print(f"  XGBoost: 平码匹配 {match_xgbs}/6")
print(f"\n仅改特码实验 (检验平码独立性):")
print(f"  V3:      平码变化 {6-match_v3ss}/6")
print(f"  XGBoost: 平码变化 {6-match_xgbss}/6")
