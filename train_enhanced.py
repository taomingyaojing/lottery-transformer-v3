#!/usr/bin/env python3
"""
彩票Transformer v3 — 强化训练版
利用4核CPU + 7.5G内存 + 更优超参 + 更多数据增强
"""
import csv, json, os, sys, random, time
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

# 利用多核
NUM_WORKERS = 3  # 留一核给系统
BATCH_SIZE = 128  # 加大batch，加快训练
NUM_EPOCHS = 50   # 更多epochs
LR = 5e-4         # 稍高学习率
WEIGHT_DECAY = 1e-4
SEQ_LEN = 8
VEC_DIM = 343

device = torch.device("cpu")
print(f"设备: {device}")
print(f"CPU核心: {os.cpu_count()}, 工作线程: {NUM_WORKERS}")
print(f"Batch: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}, LR: {LR}")
sys.stdout.flush()

# ====== 读取数据 ======
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
def encode_draw(nums, spc=None):
    vec = np.zeros(VEC_DIM, dtype=np.float32)
    for i, n in enumerate(nums):
        if 1 <= n <= 49: vec[i * 49 + (n - 1)] = 1.0
    if spc and 1 <= spc <= 49:
        vec[6 * 49 + (spc - 1)] = 1.0
    return vec

# ====== 高级数据增强 ======
print("生成增强数据...")
sys.stdout.flush()

all_samples = []

# 1. 基础滑动窗口
for i in range(SEQ_LEN, len(rows)):
    ctx_nums = [rows[j][0] for j in range(i-SEQ_LEN, i)]
    ctx_spc = [rows[j][1] for j in range(i-SEQ_LEN, i)]
    x_vecs = [encode_draw(n, s) for n, s in zip(ctx_nums, ctx_spc)]
    y_vec = encode_draw(rows[i][0], rows[i][1])
    all_samples.append((np.array(x_vecs), y_vec))
print(f"  基础窗口: {len(all_samples)}")

# 2. 跳步窗口 (跳过1-5步)
for skip in [1, 2, 3, 4, 5]:
    for i in range(SEQ_LEN, len(rows) - skip):
        if random.random() > 0.3: continue  # 30%采样减少冗余
        ctx_nums = [rows[j][0] for j in range(i-SEQ_LEN, i)]
        ctx_spc = [rows[j][1] for j in range(i-SEQ_LEN, i)]
        x_vecs = [encode_draw(n, s) for n, s in zip(ctx_nums, ctx_spc)]
        y_vec = encode_draw(rows[i+skip][0], rows[i+skip][1])
        all_samples.append((np.array(x_vecs), y_vec))
print(f"  跳步窗口后: {len(all_samples)}")

# 3. 噪声注入
for i in range(SEQ_LEN, len(rows)-1):
    if random.random() < 0.08:  # 8%概率
        ctx_nums = [rows[j][0] for j in range(i-SEQ_LEN, i)]
        ctx_spc = [rows[j][1] for j in range(i-SEQ_LEN, i)]
        x_vecs = np.array([encode_draw(n, s) for n, s in zip(ctx_nums, ctx_spc)])
        x_vecs += np.random.randn(*x_vecs.shape) * 0.02
        x_vecs = np.clip(x_vecs, 0, 1)
        y_vec = encode_draw(rows[i][0], rows[i][1])
        all_samples.append((x_vecs, y_vec))
print(f"  噪声注入后: {len(all_samples)}")

# 4. 号码随机置换
for i in range(SEQ_LEN, len(rows)-1):
    if random.random() < 0.03:
        ctx_nums = [rows[j][0] for j in range(i-SEQ_LEN, i)]
        ctx_spc = [rows[j][1] for j in range(i-SEQ_LEN, i)]
        shuffled_ctx = []
        for nums in ctx_nums:
            shuffled = list(nums)
            random.shuffle(shuffled)
            shuffled_ctx.append(shuffled)
        x_vecs = np.array([encode_draw(n, s) for n, s in zip(shuffled_ctx, ctx_spc)])
        y_vec = encode_draw(rows[i][0], rows[i][1])
        all_samples.append((x_vecs, y_vec))
print(f"  置换后: {len(all_samples)}")

# 5. 多上下文长度
for ctx_len in [4, 6, 10, 12]:
    for i in range(ctx_len, len(rows)-1):
        if random.random() < 0.03:
            ctx_nums = [rows[j][0] for j in range(i-ctx_len, i)]
            ctx_spc = [rows[j][1] for j in range(i-ctx_len, i)]
            x_vecs = [encode_draw(n, s) for n, s in zip(ctx_nums, ctx_spc)]
            x_arr = np.array(x_vecs)
            if len(x_arr) < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - len(x_arr), VEC_DIM))
                x_arr = np.vstack([pad, x_arr])
            elif len(x_arr) > SEQ_LEN:
                x_arr = x_arr[-SEQ_LEN:]
            y_vec = encode_draw(rows[i][0], rows[i][1])
            all_samples.append((x_arr, y_vec))
print(f"  多上下文后: {len(all_samples)}")

# 6. 标签平滑模拟（对y增加小噪声）
for i in range(SEQ_LEN, len(rows)-1):
    if random.random() < 0.05:
        ctx_nums = [rows[j][0] for j in range(i-SEQ_LEN, i)]
        ctx_spc = [rows[j][1] for j in range(i-SEQ_LEN, i)]
        x_vecs = [encode_draw(n, s) for n, s in zip(ctx_nums, ctx_spc)]
        y_vec = encode_draw(rows[i][0], rows[i][1])
        # 对y加小噪声作为标签平滑
        y_vec = y_vec + np.random.uniform(0, 0.05, VEC_DIM)
        y_vec = y_vec / y_vec.sum()  # 重归一化
        all_samples.append((np.array(x_vecs), y_vec))
print(f"  标签平滑后: {len(all_samples)}")

random.shuffle(all_samples)

# 限制最大样本数（防止OOM）
MAX_AUGMENT = 300000
if len(all_samples) > MAX_AUGMENT:
    all_samples = all_samples[:MAX_AUGMENT]

print(f"  最终样本数: {len(all_samples)}")
sys.stdout.flush()

# ====== 分割 ======
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

train_loader = DataLoader(LotteryDataset(train_samples), batch_size=BATCH_SIZE, 
                          shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(LotteryDataset(val_samples), batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"训练: {len(train_samples)}, 验证: {len(val_samples)}")
sys.stdout.flush()

# ====== 模型 (强化版) ======
class LotteryTransformerV3(nn.Module):
    def __init__(self, input_dim=VEC_DIM, d_model=512, nhead=8, num_layers=8, dim_feedforward=2048):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        # 更多层 + 更宽FFN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=0.15,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 全局特征
        self.global_fc = nn.Sequential(
            nn.Linear(d_model * SEQ_LEN, 1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(512, d_model)
        )
        # 每个位置独立输出头（更深）
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 512), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.15),
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
        last = x[:, -1, :] + global_feat * 0.3
        last = self.dropout(last)
        outputs = [head(last) for head in self.output_heads]
        return torch.stack(outputs, dim=1)

model = LotteryTransformerV3()
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params/1e6:.2f}M")
sys.stdout.flush()

# ====== 训练 ======
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# 余弦退火 + 重启
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑

print(f"\n开始训练 {NUM_EPOCHS} epochs...")
print(f"  每个epoch: {len(train_loader)} batch x {BATCH_SIZE}")
sys.stdout.flush()

best_val_loss = float('inf')
history = {"train_loss": [], "val_loss": [], "val_acc": []}

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    
    # Training
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)  # [B, 7, 49]
        loss = 0
        for pos in range(7):
            loss += loss_fn(outputs[:, pos, :], batch_y[:, pos*49:(pos+1)*49])
        loss /= 7
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = 0
            for pos in range(7):
                loss += loss_fn(outputs[:, pos, :], batch_y[:, pos*49:(pos+1)*49])
                pred = outputs[:, pos, :].argmax(dim=1)
                true = batch_y[:, pos*49:(pos+1)*49].argmax(dim=1)
                val_correct += (pred == true).sum().item()
                val_total += true.size(0)
            val_loss += loss.item() / 7
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total * 100
    
    scheduler.step()
    
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(val_acc)
    
    epoch_time = time.time() - epoch_start
    total_time = time.time() - start_time
    
    print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
          f"train: {avg_train_loss:.4f} | val: {avg_val_loss:.4f} | "
          f"acc: {val_acc:.2f}% | "
          f"lr: {optimizer.param_groups[0]['lr']:.6f} | "
          f"{epoch_time:.0f}s | total: {total_time:.0f}s")
    sys.stdout.flush()
    
    # 保存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")
        print(f"  ✅ 新最佳模型已保存 (val_loss={avg_val_loss:.4f})")
        sys.stdout.flush()

# 最终保存
torch.save(model.state_dict(), f"{OUTPUT_DIR}/final_model.pth")

# 保存训练历史
history["epochs"] = NUM_EPOCHS
history["best_val_loss"] = best_val_loss
history["total_time"] = time.time() - start_time
history["total_params"] = total_params
with open(f"{OUTPUT_DIR}/training_history.json", "w") as f:
    json.dump(history, f, indent=2)

total_time = time.time() - start_time
print(f"\n🎉 训练完成！总耗时: {total_time:.0f}s ({total_time/60:.1f}分钟)")
print(f"   最佳val_loss: {best_val_loss:.4f}")
print(f"   模型保存至: {OUTPUT_DIR}/")
print(f"   模型大小: {os.path.getsize(f'{OUTPUT_DIR}/best_model.pth')/1024/1024:.1f}MB")
print(f"   参数量: {total_params/1e6:.2f}M")
