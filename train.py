#!/usr/bin/env python3
"""
彩票Transformer v3 — 数据增强训练
用2240期历史数据 + 滑动窗口增强，生成10万+训练样本
"""
import csv, json, os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

OUTPUT_DIR = "./model_output"
DATA_PATH = "./data/lottery_history.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cpu")
print(f"设备: {device}")
sys.stdout.flush()

# ====== 读取原始数据 ======
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

print(f"原始数据: {len(rows)}期")
sys.stdout.flush()

# ====== 编码函数 ======
VEC_DIM = 343  # 7*49

def encode_draw(nums, spc=None):
    vec = np.zeros(VEC_DIM, dtype=np.float32)
    for i, n in enumerate(nums):
        if 1 <= n <= 49: vec[i * 49 + (n - 1)] = 1.0
    if spc and 1 <= spc <= 49: vec[6 * 49 + (spc - 1)] = 1.0
    return vec

# ====== 数据增强 ======
print("生成增强数据...")
sys.stdout.flush()

all_samples = []  # (x_seq, y_vec)

SEQ_LEN = 8
MAX_AUGMENT = 200000  # 最多生成20万样本

# 1. 基础滑动窗口 (所有相邻序列)
for i in range(SEQ_LEN, len(rows)):
    ctx_nums = [rows[j][0] for j in range(i-SEQ_LEN, i)]
    ctx_spc = [rows[j][1] for j in range(i-SEQ_LEN, i)]
    x_vecs = [encode_draw(n, s) for n, s in zip(ctx_nums, ctx_spc)]
    y_vec = encode_draw(rows[i][0], rows[i][1])
    all_samples.append((np.array(x_vecs), y_vec))

print(f"  基础滑动窗口: {len(all_samples)}个")
sys.stdout.flush()

# 2. 跳步窗口 (跳过1-3步, 模拟缺失数据)
for skip in [1, 2, 3]:
    for i in range(SEQ_LEN, len(rows) - skip):
        ctx_nums = [rows[j][0] for j in range(i-SEQ_LEN, i)]
        ctx_spc = [rows[j][1] for j in range(i-SEQ_LEN, i)]
        x_vecs = [encode_draw(n, s) for n, s in zip(ctx_nums, ctx_spc)]
        y_vec = encode_draw(rows[i+skip][0], rows[i+skip][1])
        all_samples.append((np.array(x_vecs), y_vec))

print(f"  加跳步窗口后: {len(all_samples)}个")
sys.stdout.flush()

# 3. 噪声注入 (对输入添加小噪声)
for i in range(SEQ_LEN, len(rows)-1):
    if random.random() < 0.05:  # 5%概率
        ctx_nums = [rows[j][0] for j in range(i-SEQ_LEN, i)]
        ctx_spc = [rows[j][1] for j in range(i-SEQ_LEN, i)]
        x_vecs = np.array([encode_draw(n, s) for n, s in zip(ctx_nums, ctx_spc)])
        # 添加高斯噪声
        x_vecs += np.random.randn(*x_vecs.shape) * 0.01
        x_vecs = np.clip(x_vecs, 0, 1)
        y_vec = encode_draw(rows[i][0], rows[i][1])
        all_samples.append((x_vecs, y_vec))

# 4. 顺序打乱增强 (对每期内的号码随机置换, 保持数据分布)
for i in range(SEQ_LEN, len(rows)-1):
    if random.random() < 0.02:  # 2%概率
        ctx_nums = [rows[j][0] for j in range(i-SEQ_LEN, i)]
        ctx_spc = [rows[j][1] for j in range(i-SEQ_LEN, i)]
        # 随机打乱每期内的号码顺序
        shuffled_ctx = []
        for nums in ctx_nums:
            shuffled = list(nums)
            random.shuffle(shuffled)
            shuffled_ctx.append(shuffled)
        x_vecs = np.array([encode_draw(n, s) for n, s in zip(shuffled_ctx, ctx_spc)])
        y_vec = encode_draw(rows[i][0], rows[i][1])
        all_samples.append((x_vecs, y_vec))

# 5. 多上下文长度 (3,5,10步上下文)
for ctx_len in [3, 5, 10]:
    for i in range(ctx_len, len(rows)-1):
        if random.random() < 0.05:  # 5%采样
            ctx_nums = [rows[j][0] for j in range(i-ctx_len, i)]
            ctx_spc = [rows[j][1] for j in range(i-ctx_len, i)]
            x_vecs = [encode_draw(n, s) for n, s in zip(ctx_nums, ctx_spc)]
            # 补足到SEQ_LEN
            if ctx_len < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - ctx_len, VEC_DIM))
                x_vecs = np.vstack([pad, x_vecs])
            elif ctx_len > SEQ_LEN:
                x_vecs = x_vecs[-SEQ_LEN:]
            y_vec = encode_draw(rows[i][0], rows[i][1])
            all_samples.append((np.array(x_vecs), y_vec))

random.shuffle(all_samples)

# 限制最大样本数
if len(all_samples) > MAX_AUGMENT:
    all_samples = all_samples[:MAX_AUGMENT]

print(f"  增强后总样本: {len(all_samples)}个")
sys.stdout.flush()

# 按时间分割 (前80%训练, 20%验证)
split = int(len(all_samples) * 0.8)
train_samples = all_samples[:split]
val_samples = all_samples[split:]

class LotteryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)

train_loader = DataLoader(LotteryDataset(train_samples), batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(LotteryDataset(val_samples), batch_size=64, shuffle=False, num_workers=0)

print(f"训练: {len(train_samples)}, 验证: {len(val_samples)}")
sys.stdout.flush()

# ====== 模型 (加深加宽) ======
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
        outputs = [head(last) for head in self.output_heads]
        return torch.stack(outputs, dim=1)

model = LotteryTransformerV3()
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params/1e6:.2f}M")
sys.stdout.flush()

# ====== 训练 ======
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
loss_fn = nn.CrossEntropyLoss()

NUM_EPOCHS = 30
best_val_loss = float('inf')
history = []

print(f"\n开始训练 {NUM_EPOCHS} epochs, batch=64...")
print(f"每个epoch步数: {len(train_loader)}")
sys.stdout.flush()

for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        y_r = y.view(-1, 7, 49)
        optimizer.zero_grad()
        pred = model(x)
        # 特码权重加倍
        losses = [loss_fn(pred[:, p, :], y_r[:, p, :]) for p in range(7)]
        loss = (sum(losses[:6]) + losses[6] * 2) / 8  # 特码×2权重
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    
    # Eval
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            y_r = y.view(-1, 7, 49)
            pred = model(x)
            losses = [loss_fn(pred[:, p, :], y_r[:, p, :]) for p in range(7)]
            loss = (sum(losses[:6]) + losses[6] * 2) / 8
            val_loss += loss.item()
            val_correct += (pred.argmax(-1) == y_r.argmax(-1)).sum().item()
            val_total += pred.shape[0] * 7
    
    scheduler.step()
    
    train_loss_avg = train_loss / len(train_loader)
    val_loss_avg = val_loss / len(val_loader)
    val_acc = val_correct / val_total * 100
    
    if (epoch+1) % 2 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:2d}: train_loss={train_loss_avg:.4f} | val_loss={val_loss_avg:.4f} val_acc={val_acc:.2f}%")
        sys.stdout.flush()
    
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")
        print(f"  -> 新最佳 (val_loss={best_val_loss:.4f})")
        sys.stdout.flush()

print(f"\n训练完成！最佳验证loss={best_val_loss:.4f}")
sys.stdout.flush()

# ====== 回测 ======
print("\n=== 滚动回测 (最近200期) ===")
sys.stdout.flush()

model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pth", map_location=device, weights_only=True))
model.eval()

all_vecs = np.array([encode_draw(n, s) for n, s in rows])
correct_top1 = [0]*7
correct_top3 = [0]*7
correct_top5 = [0]*7
total_test = 0

for i in range(len(all_vecs) - SEQ_LEN - 200, len(all_vecs) - SEQ_LEN):
    x = all_vecs[i:i+SEQ_LEN]
    y_true = all_vecs[i+SEQ_LEN].reshape(7, 49).argmax(axis=1)
    
    with torch.no_grad():
        pred = model(torch.FloatTensor(x).unsqueeze(0))
    
    probs = F.softmax(pred, dim=-1).squeeze(0).numpy()
    pred_nums = pred.argmax(-1).squeeze(0).numpy()
    
    for p in range(7):
        if pred_nums[p] == y_true[p]: correct_top1[p] += 1
        top3 = np.argsort(probs[p])[-3:]
        if y_true[p] in top3: correct_top3[p] += 1
        top5 = np.argsort(probs[p])[-5:]
        if y_true[p] in top5: correct_top5[p] += 1
    
    total_test += 1

print(f"回测{total_test}期:")
for p in range(7):
    label = f"位置{p+1}" if p < 6 else "特码"
    print(f"  {label}: top1={correct_top1[p]}({correct_top1[p]/total_test*100:.1f}%) top3={correct_top3[p]}({correct_top3[p]/total_test*100:.1f}%) top5={correct_top5[p]}({correct_top5[p]/total_test*100:.1f}%)")
sys.stdout.flush()

# ====== 最终预测 ======
print("\n=== 最终预测 ===")
sys.stdout.flush()

# 多窗口投票
windows = [3, 5, 8, 10, 12]
votes = np.zeros((7, 49))

for ctx in windows:
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

final = votes.argmax(axis=1)
print(f"  平码: {[int(final[i])+1 for i in range(6)]}")
print(f"  特码: {int(final[6])+1}")
sys.stdout.flush()

print("\n  每位置Top5:")
for p in range(7):
    label = f"位置{p+1}" if p < 6 else "特码"
    top5 = np.argsort(votes[p])[-5:][::-1]
    top5_str = ", ".join([f"{int(idx)+1}({votes[p][idx]:.3f})" for idx in top5])
    print(f"    {label}: {top5_str}")
sys.stdout.flush()

# ====== 保存结果 ======
output = {
    "model": "LotteryTransformerV3_Augmented",
    "params_M": round(total_params/1e6, 2),
    "training": {
        "epochs": NUM_EPOCHS,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "augmentation_methods": ["sliding_window", "skip_window", "noise_injection", "shuffle", "multi_context"],
        "best_val_loss": round(best_val_loss, 4),
    },
    "backtest_200": {
        "pos1_top1": f"{correct_top1[0]}/{total_test}({correct_top1[0]/total_test*100:.1f}%)",
        "pos2_top1": f"{correct_top1[1]}/{total_test}({correct_top1[1]/total_test*100:.1f}%)",
        "pos3_top1": f"{correct_top1[2]}/{total_test}({correct_top1[2]/total_test*100:.1f}%)",
        "pos4_top1": f"{correct_top1[3]}/{total_test}({correct_top1[3]/total_test*100:.1f}%)",
        "pos5_top1": f"{correct_top1[4]}/{total_test}({correct_top1[4]/total_test*100:.1f}%)",
        "pos6_top1": f"{correct_top1[5]}/{total_test}({correct_top1[5]/total_test*100:.1f}%)",
        "spc_top1": f"{correct_top1[6]}/{total_test}({correct_top1[6]/total_test*100:.1f}%)",
        "spc_top3": f"{correct_top3[6]}/{total_test}({correct_top3[6]/total_test*100:.1f}%)",
        "spc_top5": f"{correct_top5[6]}/{total_test}({correct_top5[6]/total_test*100:.1f}%)",
    },
    "prediction": {
        "nums": [int(final[i])+1 for i in range(6)],
        "spc": int(final[6])+1,
        "position_top5": {
            f"pos{p+1}" if p < 6 else "spc": [int(np.argsort(votes[p])[-5:][::-1][i])+1 for i in range(5)]
            for p in range(7)
        }
    }
}

with open(f"{OUTPUT_DIR}/v3_result.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n结果保存到: {OUTPUT_DIR}/v3_result.json")
sys.stdout.flush()
