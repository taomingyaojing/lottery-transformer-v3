#!/usr/bin/env python3
"""
V3编码噪声实验 — 上期数字加0.01，看one-hot编码强度变化对预测的影响
"""
import csv, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DATA_PATH = "/home/ubuntu/lottery_bert_research/data/lottery_all_years_updated_20260423.csv"
MODEL_DIR = "/home/ubuntu/lottery_bert_research/ft_model_v3"
SEQ_LEN = 8; VEC_DIM = 343

# 数据
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

# V3模型
class LotteryTransformerV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(343, 512)
        self.pos_encoder = nn.Parameter(torch.randn(1, 8, 512) * 0.02)
        el = nn.TransformerEncoderLayer(512, 8, 1024, 0.2, 'gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(el, 6)
        self.global_fc = nn.Sequential(nn.Linear(512*8, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 512))
        self.output_heads = nn.ModuleList([nn.Sequential(nn.Linear(512,256), nn.GELU(), nn.Dropout(0.15), nn.Linear(256,128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128,49)) for _ in range(7)])
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        B = x.shape[0]
        x = self.input_proj(x) + self.pos_encoder
        x = self.transformer(x)
        gf = self.global_fc(x.reshape(B, -1))
        last = x[:, -1, :] + gf * 0.2
        last = self.dropout(last)
        return torch.stack([h(last) for h in self.output_heads], dim=1)

device = torch.device("cpu")
model = LotteryTransformerV3()
model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model.pth", map_location=device, weights_only=True))
model.eval()

def encode_draw(nums, spc=None, noise_map=None):
    """noise_map: {位置: 编码强度} 覆盖默认1.0"""
    vec = np.zeros(VEC_DIM, dtype=np.float32)
    for i, n in enumerate(nums):
        if 1 <= n <= 49:
            val = 1.0
            if noise_map and i in noise_map:
                val = noise_map[i]
            vec[i * 49 + (n - 1)] = val
    if spc and 1 <= spc <= 49:
        vec[6 * 49 + (spc - 1)] = 1.0
    return vec

def predict(rows_data, noise_map=None):
    votes = np.zeros((7, 49))
    for ctx in [3, 5, 8, 10]:
        if ctx > len(rows_data): continue
        seq = []
        for i in range(ctx):
            idx = -ctx + i
            wm = noise_map if i == ctx-1 else None
            seq.append(encode_draw(rows_data[idx][0], rows_data[idx][1], wm))
        x = np.array(seq)
        if ctx < 8: x = np.vstack([np.zeros((8-ctx, 343)), x])
        elif ctx > 8: x = x[-8:]
        with torch.no_grad():
            p = model(torch.FloatTensor(x).unsqueeze(0))
        votes += F.softmax(p, dim=-1).squeeze(0).numpy()
    return votes

# 找到上期每个数字在序列中的位置
last = rows[-1]
print(f"上期: {last[0]} + T{last[1]}")
print()

# 原始预测
orig_votes = predict(rows)
orig = orig_votes.argmax(axis=1)
orig_str = f"[{', '.join(f'{int(orig[i])+1:02d}' for i in range(6))}] + T{int(orig[6])+1:02d}"
print(f"原始预测:     {orig_str}")

# 实验: 逐个位置加噪声(0.01)
print("\n逐个位置上期数字编码强度微调 (+0.01):")
print("-" * 60)

all_agree = True
for pos, num in enumerate(last[0]):
    noise_map = {pos: 1.01}  # 原始1.0 + 0.01
    noise_votes = predict(rows, noise_map=noise_map)
    noise = noise_votes.argmax(axis=1)
    
    changes = sum(1 for i in range(7) if orig[i] != noise[i])
    noise_str = f"[{', '.join(f'{int(noise[i])+1:02d}' for i in range(6))}] + T{int(noise[6])+1:02d}"
    
    if changes > 0:
        all_agree = False
        print(f"位置{pos+1}({num:02d})+0.01: {noise_str}  ← 变了{changes}/7!")
    else:
        print(f"位置{pos+1}({num:02d})+0.01: {noise_str}  没变")

# 特码也来
spc_noise_map = {f"spc": 1.01}
def predict_spc_noise(rows_data):
    votes = np.zeros((7, 49))
    for ctx in [3, 5, 8, 10]:
        if ctx > len(rows_data): continue
        seq = []
        for i in range(ctx):
            idx = -ctx + i
            ns = rows_data[idx][0]
            sp = rows_data[idx][1]
            # 对最后一期的特码编码加0.01
            if i == ctx-1:
                vec = encode_draw(ns, sp, noise_map={})
                # 直接在spc位置加0.01
                if sp and 1 <= sp <= 49:
                    vec[6*49 + (sp-1)] = 1.01
                seq.append(vec)
            else:
                seq.append(encode_draw(ns, sp))
        x = np.array(seq)
        if ctx < 8: x = np.vstack([np.zeros((8-ctx, 343)), x])
        elif ctx > 8: x = x[-8:]
        with torch.no_grad():
            p = model(torch.FloatTensor(x).unsqueeze(0))
        votes += F.softmax(p, dim=-1).squeeze(0).numpy()
    return votes

noise_votes = predict_spc_noise(rows)
noise = noise_votes.argmax(axis=1)
changes = sum(1 for i in range(7) if orig[i] != noise[i])
noise_str = f"[{', '.join(f'{int(noise[i])+1:02d}' for i in range(6))}] + T{int(noise[6])+1:02d}"
print(f"特码(T32)+0.01:  {noise_str}  {'← 变了!' if changes > 0 else '没变'}")

print()
if all_agree:
    print("✅ V3对0.01的编码波动完全不敏感 — 所有位置预测没变")
else:
    print("⚠️ 有位置对0.01的波动有反应")
