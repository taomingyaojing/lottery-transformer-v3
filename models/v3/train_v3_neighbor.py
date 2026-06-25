#!/usr/bin/env python3
"""
🦦 V3增强版 — 邻域先验感知模型 (Ensemble Prediction with Neighborhood Prior)

核心改进:
1. 特征层: 输入中编码"上一期数字集"的邻域热度（49维高斯衰减）
2. 后处理层: softmax输出后用邻域先验修正（+20%权重给上期数字±3范围内的号码）
3. 训练: 仅分类损失，但推理加权融合邻域知识

关键洞察: 平码排序破坏了位置对应关系，所以不做offset预测，
而是在推理阶段直接用"集合邻域"修正输出
"""
import csv, json, os, sys, math, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cpu")

# ============ 配置 ============
OUTPUT_DIR = "./models/v3"
DATA_PATH = "./data/lottery_history.csv"
SEQ_LEN = 8
VEC_DIM = 343
NEI_DIM = 49
TOTAL_DIM = VEC_DIM + NEI_DIM + NEI_DIM  # 343 + 49(邻域距离) + 49(领域衰减)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ 数据加载 ============
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

print(f"📊 数据: {len(rows)} 期")

# ============ 编码函数 ============
def encode_draw(nums, spc=None):
    """基础 one-hot 343维"""
    vec = np.zeros(VEC_DIM, dtype=np.float32)
    for i, n in enumerate(nums):
        if 1 <= n <= 49: vec[i * 49 + (n - 1)] = 1.0
    if spc and 1 <= spc <= 49: vec[6 * 49 + (spc - 1)] = 1.0
    return vec

def encode_neighborhood_mask(prev_nums, prev_spc):
    """
    邻域掩码编码: 49维
    对每个数字1-49，编码它到上一期数字集的最小距离
    使用高斯衰减: exp(-dist^2 / (2*sigma^2))
    sigma=2.5 使得±3范围内有明显激活
    """
    nei = np.zeros(49, dtype=np.float32)
    all_prev = list(prev_nums)
    if prev_spc: all_prev.append(prev_spc)
    
    sigma = 2.5
    for n in range(1, 50):
        min_dist = min(abs(n - p) for p in all_prev)
        nei[n-1] = math.exp(-(min_dist**2) / (2 * sigma**2))
    
    return nei

def encode_neighborhood_distance(prev_nums, prev_spc):
    """
    距离编码: 49维，直接存储最小距离
    """
    nei = np.zeros(49, dtype=np.float32)
    all_prev = list(prev_nums)
    if prev_spc: all_prev.append(prev_spc)
    
    for n in range(1, 50):
        min_dist = min(abs(n - p) for p in all_prev)
        nei[n-1] = min_dist / 24.0  # 归一化到0-1
    
    return nei

def prepare_sample(idx):
    """准备一个训练样本"""
    if idx < SEQ_LEN + 1:
        return None
    
    # 输入序列的 one-hot
    x_vecs = []
    for i in range(idx - SEQ_LEN, idx):
        nums, spc = rows[i]
        x_vecs.append(encode_draw(nums, spc))
    x = np.stack(x_vecs)  # (SEQ_LEN, 343)
    
    # 邻域特征（最后一期的上一期）
    prev_nums, prev_spc = rows[idx-1]
    nei_mask = encode_neighborhood_mask(prev_nums, prev_spc)  # (49,)
    nei_dist = encode_neighborhood_distance(prev_nums, prev_spc)  # (49,)
    
    # 将邻域特征复制到每个时间步
    nei_mask_stacked = np.tile(nei_mask, (SEQ_LEN, 1))  # (8, 49)
    nei_dist_stacked = np.tile(nei_dist, (SEQ_LEN, 1))  # (8, 49)
    
    # 拼接
    combined = np.concatenate([x, nei_mask_stacked, nei_dist_stacked], axis=1)  # (8, 441)
    
    # 目标（下一期）
    target_nums, target_spc = rows[idx]
    target = np.zeros(7, dtype=np.int64)
    for i, n in enumerate(target_nums):
        target[i] = n - 1
    target[6] = target_spc - 1
    
    return {
        "x": torch.FloatTensor(combined),
        "target": torch.LongTensor(target),
        "nei_mask": torch.FloatTensor(nei_mask),
    }

class LotteryDataset(Dataset):
    def __init__(self, start, end):
        self.indices = list(range(start, end))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        idx = self.indices[i]
        sample = prepare_sample(idx)
        if sample is None:
            return self.__getitem__((i + 1) % len(self))
        return sample

# ============ 模型 ============
class NeighborhoodEnhancedTransformer(nn.Module):
    def __init__(self, input_dim=441, d_model=512, nhead=8, num_layers=6, dim_feedforward=1024):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=0.2,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.global_fc = nn.Sequential(
            nn.Linear(d_model * SEQ_LEN, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, d_model)
        )
        
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(128, 49)
            ) for _ in range(7)
        ])
        
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

# ============ 推理增强函数 ============
def predict_with_neighborhood(model, logits, nei_mask, alpha=0.25):
    """
    推理时用邻域先验修正softmax输出
    
    Args:
        logits: (7, 49) 每个位置的分类logits
        nei_mask: (49,) 邻域掩码（高斯衰减）
        alpha: 邻域先验权重
    """
    probs = F.softmax(logits, dim=-1)  # (7, 49)
    
    # 对每个位置，将邻域先验加权融合
    enhanced = (1 - alpha) * probs + alpha * nei_mask.unsqueeze(0)  # (7, 49)
    
    return enhanced

def predict_with_voting(model, rows, num_contexts=[3, 5, 8, 10], alpha=0.25):
    """多窗口voting + 邻域先验融合"""
    votes = np.zeros((7, 49))
    
    for ctx in num_contexts:
        if ctx >= len(rows): continue
        # 构建输入
        x_vecs = []
        for i in range(len(rows) - ctx, len(rows)):
            nums, spc = rows[i]
            x_vecs.append(encode_draw(nums, spc))
        
        # pad到SEQ_LEN
        x = np.array(x_vecs)
        if ctx < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - ctx, VEC_DIM))
            x = np.vstack([pad, x])
        elif ctx > SEQ_LEN:
            x = x[-SEQ_LEN:]
        
        # 邻域特征（最后一期的上一期）
        prev_nums, prev_spc = rows[-ctx-1] if ctx < len(rows) else rows[-2]
        nei_mask = encode_neighborhood_mask(prev_nums, prev_spc)
        nei_dist = encode_neighborhood_distance(prev_nums, prev_spc)
        nei_mask_stacked = np.tile(nei_mask, (SEQ_LEN, 1))
        nei_dist_stacked = np.tile(nei_dist, (SEQ_LEN, 1))
        combined = np.concatenate([x, nei_mask_stacked, nei_dist_stacked], axis=1)
        
        with torch.no_grad():
            logits = model(torch.FloatTensor(combined).unsqueeze(0))
            enhanced = predict_with_neighborhood(model, logits.squeeze(0), torch.FloatTensor(nei_mask), alpha)
        
        votes += enhanced.numpy()
    
    # 最佳组合（6个不重复平码 + 1个特码）
    votes_flat = votes[:6].sum(axis=0)
    best = np.argsort(votes_flat)[-15:][::-1]
    combo = []
    for b in best:
        if b + 1 not in combo:
            combo.append(b + 1)
        if len(combo) == 6: break
    for n in range(1, 50):
        if n not in combo and len(combo) < 6:
            combo.append(n)
    
    spc = int(votes[6].argmax()) + 1
    
    return combo, spc, votes

# ============ 训练 ============
print("\n🔧 准备训练数据...")
train_start = SEQ_LEN + 1
val_start = len(rows) - 50

train_ds = LotteryDataset(train_start, val_start)
val_ds = LotteryDataset(val_start, len(rows))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

print(f"  训练: {len(train_ds)} 样本")
print(f"  验证: {len(val_ds)} 样本")

model = NeighborhoodEnhancedTransformer()
print(f"  参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_val_loss = float('inf')
EPOCHS = 40

print("\n🚀 训练中...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    ntrain = 0
    for batch in train_loader:
        x = batch["x"]
        t = batch["target"]
        
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 49), t.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
        ntrain += 1
    
    model.eval()
    val_loss = 0
    nval = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"]
            t = batch["target"]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 49), t.reshape(-1))
            val_loss += loss.item()
            nval += 1
    
    train_loss /= ntrain
    val_loss /= nval
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")
    
    print(f"  Epoch {epoch+1:2d}: train={train_loss:.4f} val={val_loss:.4f} {'✅' if val_loss == best_val_loss else ''}")
    
    scheduler.step()

torch.save(model.state_dict(), f"{OUTPUT_DIR}/final_model.pth")
print(f"\n✅ 训练完成! 最佳模型: {OUTPUT_DIR}/best_model.pth (val_loss={best_val_loss:.4f})")

# ============ 推理测试 ============
print("\n🔮 推理测试（最近10期留出）...")
model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pth", map_location=device, weights_only=True))
model.eval()

# 对比: 纯V3 vs V3+邻域
for alpha in [0.0, 0.15, 0.25, 0.35]:
    hits = 0
    total = 0
    for idx in range(len(rows) - 10, len(rows)):
        sample = prepare_sample(idx)
        if sample is None: continue
        
        x = sample["x"].unsqueeze(0)
        nei_mask = sample["nei_mask"]
        cls_t = sample["target"].numpy()
        
        with torch.no_grad():
            logits = model(x)
        
        if alpha > 0:
            enhanced = predict_with_neighborhood(model, logits.squeeze(0), nei_mask, alpha)
            pred = enhanced.argmax(dim=-1).numpy()
        else:
            pred = logits.squeeze(0).argmax(dim=-1).numpy()
        
        hits += sum(1 for i in range(7) if pred[i] == cls_t[i])
        total += 7
    
    print(f"  α={alpha:.2f}: 准确率 = {hits}/{total} = {hits/total*100:.2f}%")

# 最终预测
print("\n🎯 最终预测 (下一期):")
alpha_best = 0.25
nums, spc, votes = predict_with_voting(model, rows, alpha=alpha_best)
print(f"  平码: {nums}")
print(f"  特码: {spc}")
print(f"  全7: {nums + [spc]}")

# 保存
pred = {
    "engine": "V3_Neighborhood_v1",
    "date": "2026-06-25",
    "params": {"neighborhood_alpha": alpha_best, "seq_len": SEQ_LEN, "voting_contexts": [3,5,8,10]},
    "prediction": {
        "nums": nums,
        "spc": spc
    }
}
with open(f"{OUTPUT_DIR}/prediction.json", "w") as f:
    json.dump(pred, f, indent=2)
print(f"\n✅ 预测已保存: {OUTPUT_DIR}/prediction.json")
