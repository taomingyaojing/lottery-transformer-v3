#!/usr/bin/env python3
import csv, json, os, sys
from datetime import datetime
"""
V3增强推理 — 基础v3 softmax + 冷热度后处理补偿
策略:
1. 统计每个数字的近期频率（最近30/60/100期出现次数）和gap
2. 为softmax输出叠加一个冷热度偏差
3. 对热门候选优先选温号，对冷门位置做补偿
"""
import csv, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUTPUT_DIR = "./model_output_backup_20260615_140503"
DATA_PATH = "./data/lottery_history.csv"
PRED_DIR = "./predictions"
os.makedirs(PRED_DIR, exist_ok=True)

SEQ_LEN = 8
VEC_DIM = 343

# ===== 模型定义 =====
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

# ===== 加载数据 =====
print("📊 加载数据...")
rows = []
with open(DATA_PATH) as f:
    reader = csv.reader(f)
    h = next(reader)
    for r in reader:
        nums = []
        for i in [2,6,10,14,18,22]:
            if i < len(r) and r[i].strip():
                try: nums.append(int(r[i]))
                except: break
        if len(nums) != 6: continue
        spc = None
        if 26 < len(r) and r[26].strip():
            try: spc = int(r[26])
            except: pass
        rows.append((nums, spc))

total_periods = len(rows)
last10 = rows[-10:] if len(rows) >= 10 else rows
last30 = rows[-30:] if len(rows) >= 30 else rows
last60 = rows[-60:] if len(rows) >= 60 else rows

print(f"  总期数: {total_periods}, 最近60期: {len(last60)}")

# ===== 统计热冷数据 =====
def count_freq(periods):
    """统计1-49在给定期数中出现的次数（含普通号和特码）"""
    freq = np.zeros(49, dtype=np.int32)
    for nums, spc in periods:
        for n in nums:
            if 1 <= n <= 49: freq[n-1] += 1
        if spc and 1 <= spc <= 49: freq[spc-1] += 1
    return freq

freq_10 = count_freq(last10)
freq_30 = count_freq(last30)
freq_60 = count_freq(last60)

# 计算每个数字的gap（距离上次出现的期数）
def compute_gaps(rows):
    """每个数字最后一次出现的期数偏移"""
    gaps = np.full(49, len(rows), dtype=np.int32)  # 默认 = 总期数（从来没出过）
    for offset in range(len(rows)):
        idx = len(rows) - 1 - offset  # 从最新往旧
        nums, spc = rows[idx]
        for n in list(nums) + ([spc] if spc else []):
            if 1 <= n <= 49 and gaps[n-1] == len(rows):
                gaps[n-1] = offset  # 距当前期数偏移
    return gaps  # gap=0表示上一期刚出过

gaps = compute_gaps(rows)

# ===== 冷热度评分 =====
# 热号得分：最近60期频率高  => 高分
# 温号得分：近期有出现但不多 => 中等分  
# 冷号得分：gap很大但加一点补偿 => 低分（但给予额外boost）

cold_threshold = 20  # gap>=20为冷号
warm_threshold = 8   # gap<=8为热号

cold_score = np.zeros(49)
for n in range(49):
    # 基础分：基于最近60期频率（标准化到0~1）
    base = freq_60[n] / max(np.max(freq_60), 1)
    gap_penalty = min(gaps[n] / 100.0, 1.0)  # gap越大越penalty
    
    if gaps[n] >= cold_threshold:
        # 冷号：低频但有爆发潜力，给一个小boost
        cold_score[n] = base * 0.5 + 0.3 * (1.0 - gap_penalty)
    elif gaps[n] <= warm_threshold:
        # 热号：高频率
        freq_10_n = freq_10[n] / max(np.max(freq_10), 1)
        cold_score[n] = base * 0.7 + freq_10_n * 0.3
    else:
        # 温号：中间地带，适中boost
        cold_score[n] = base * 0.6 + 0.25 * (1.0 - gap_penalty * 0.5)

# normalize到0~1
cold_score = (cold_score - np.min(cold_score)) / max(np.max(cold_score) - np.min(cold_score), 1e-8)

print(f"\n📈 热度TOP10:")
hot_indices = np.argsort(cold_score)[-10:][::-1]
for i in hot_indices:
    print(f"  {i+1}: score={cold_score[i]:.3f} gap={gaps[i]} freq60={freq_60[i]} freq30={freq_30[i]} freq10={freq_10[i]}")

print(f"\n❄️ 冷号TOP10 (gap>={cold_threshold}):")
cold_indices = np.argsort(cold_score)[:10]
for i in cold_indices:
    print(f"  {i+1}: score={cold_score[i]:.3f} gap={gaps[i]} freq60={freq_60[i]}")

# ===== 加载模型 =====
print("\n🔬 加载V3模型...")
device = torch.device("cpu")
model = LotteryTransformerV3()
model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pth", map_location=device, weights_only=True))
model.eval()
print("  V3模型加载成功!")

def encode_draw(nums, spc=None):
    vec = np.zeros(VEC_DIM, dtype=np.float32)
    for i, n in enumerate(nums):
        if 1 <= n <= 49: vec[i * 49 + (n - 1)] = 1.0
    if spc and 1 <= spc <= 49: vec[6 * 49 + (spc - 1)] = 1.0
    return vec

# ===== V3原始预测 =====
print("\n🔮 V3基础预测 (多窗口voting)...")
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
        pred = model(torch.FloatTensor(x_vecs).unsqueeze(0))
    votes += F.softmax(pred, dim=-1).squeeze(0).numpy()

# ===== 后处理：软融合冷热度补偿 =====
# 对每个位置，将cold_score以权重alpha融合到softmax输出中
# 位置1-6（普通号）和特码用不同的alpha
alpha = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.20]  # 特码给更高补偿权重
enhanced = np.zeros_like(votes)
for p in range(7):
    pure_vote = votes[p]
    # 融合
    blended = (1 - alpha[p]) * pure_vote + alpha[p] * cold_score
    enhanced[p] = blended

# ===== 构建最终推荐 =====
# 原始v3
raw_final = votes.argmax(axis=1)
# 增强版
enh_final = enhanced.argmax(axis=1)

# 保证不重复（普通号6个）
def best_combo(scores, top_n=15, pick=6):
    """从enhanced score中选择最好的6个不重复数字"""
    scores_flat = scores[:6].sum(axis=0)  # 并合6个位置
    best = np.argsort(scores_flat)[-top_n:][::-1]
    combo = []
    for b in best:
        if b + 1 not in combo:
            combo.append(b + 1)
        if len(combo) == pick:
            break
    # 如果不够6个
    for n in range(1, 50):
        if n not in combo and len(combo) < pick:
            combo.append(n)
    return sorted(combo)

enh_ord = best_combo(enhanced, top_n=15, pick=6)
enh_spc = int(enhanced[6].argmax()) + 1

# ===== 输出 =====
print("\n" + "="*60)
print("🔬 V3+ 增强预测 (冷热度后处理补偿)")
print("="*60)
print(f"\n📅 上期: {rows[-1][0]} + T{rows[-1][1]}")
print(f"\n--- V3原始预测 ---")
print(f"  平码: {[int(raw_final[i])+1 for i in range(6)]}")
print(f"  特码: {int(raw_final[6])+1}")

print(f"\n--- V3增强版 (α={alpha[0]}) ---")
print(f"  平码: {enh_ord}")
print(f"  特码: {enh_spc}")
print(f"  全7: {enh_ord + [enh_spc]}")

# 显示每个位置的Top5（融合后）
print(f"\n📊 每位置Top5 (融合后):")
for p in range(7):
    label = f"位置{p+1}" if p < 6 else "特码"
    top5 = np.argsort(enhanced[p])[-5:][::-1]
    top5_str = ", ".join([f"{int(idx)+1}({enhanced[p][idx]:.3f})" for idx in top5])
    print(f"  {label}: {top5_str}")

# 显示哪些号码是因为冷热补偿被"拉上来"的
print(f"\n🔄 冷热度补偿效果:")
for p in range(7):
    label = f"位置{p+1}" if p < 6 else "特码"
    raw_top3 = set(np.argsort(votes[p])[-3:][::-1])
    enh_top3 = set(np.argsort(enhanced[p])[-3:][::-1])
    newbies = enh_top3 - raw_top3
    if newbies:
        print(f"  {label} 新进入Top3: {[int(n)+1 for n in newbies]}")
        for n in newbies:
            print(f"    {int(n)+1}: v3原始={votes[p][n]:.4f}, 冷热度分={cold_score[n]:.4f}, 融合={enhanced[p][n]:.4f}")

# 保存
date_str = datetime.now().strftime("%Y%m%d")
pred = {
    "engine": "ft_v3_enhanced",
    "date": "2026-06-15",
    "raw": {
        "ord": [int(raw_final[i])+1 for i in range(6)],
        "spc": int(raw_final[6])+1
    },
    "enhanced": {
        "ord": [int(x) for x in enh_ord],
        "spc": int(enh_spc)
    },
    "cold_top10": [int(i)+1 for i in hot_indices],
    "gaps": [int(g) for g in gaps],
    "params": {"alpha": [float(a) for a in alpha], "cold_threshold": int(cold_threshold)}
}
pred_path = f"{PRED_DIR}/prediction_{date_str}_v3_enhanced.json"
with open(pred_path, "w") as f:
    json.dump(pred, f, indent=2)
print(f"\n✅ 已保存: {pred_path}")
print(f"⏱️ 完成")
