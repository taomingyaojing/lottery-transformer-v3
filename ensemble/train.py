#!/usr/bin/env python3
"""
多模型集成训练 v1
利用远程机4核并行训练4个不同结构的模型
最后通过投票集成提高预测准确率
"""
import csv, json, os, sys, random, time, math, threading
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

OUTPUT_DIR = "."
DATA_PATH = "../data/lottery_history.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cpu")
NUM_WORKERS = 2
BATCH_SIZE = 128
EPOCHS = 40
VEC_DIM = 343
SEQ_LEN = 8

print(f"设备: {device}")
print(f"CPU核心: {os.cpu_count()}, 工作线程: {NUM_WORKERS}")
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
        rows.append((tuple(nums), spc))

print(f"原始数据: {len(rows)}期")
sys.stdout.flush()

# ====== 增强特征编码（更丰富的特征） ======
def encode_draw_v2(nums, spc, prev_nums_list=None):
    """
    增强编码：343基础 + 额外特征
    返回 (base_vec, extra_vec) 
    base_vec: 343维 one-hot
    extra_vec: 额外特征向量
    """
    # 基础one-hot
    vec = np.zeros(VEC_DIM, dtype=np.float32)
    for i, n in enumerate(nums):
        if 1 <= n <= 49: vec[i * 49 + (n - 1)] = 1.0
    if spc and 1 <= spc <= 49:
        vec[6 * 49 + (spc - 1)] = 1.0
    
    # 额外特征
    extra = np.zeros(32, dtype=np.float32)
    
    # 1. 号码和
    extra[0] = sum(nums) / 294.0  # 归一化 (6*49=294)
    
    # 2. 奇偶比
    odd = sum(1 for n in nums if n % 2 == 1)
    extra[1] = odd / 6.0
    
    # 3. 大小比 (>=25为大)
    big = sum(1 for n in nums if n >= 25)
    extra[2] = big / 6.0
    
    # 4. 质数个数
    primes = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47}
    extra[3] = sum(1 for n in nums if n in primes) / 6.0
    
    # 5. 跨度（最大值-最小值）
    extra[4] = (max(nums) - min(nums)) / 48.0
    
    # 6. AC值（不同差值的个数）
    diffs = set()
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            diffs.add(abs(nums[i] - nums[j]))
    extra[5] = len(diffs) / 15.0
    
    # 7-10. 与历史频率的关系（冷热号）
    if prev_nums_list and len(prev_nums_list) > 20:
        recent = [n for draw in prev_nums_list[-20:] for n in draw[0]]
        freq = Counter(recent)
        extra[7] = freq.get(nums[0], 0) / 20.0
        extra[8] = sum(freq.get(n, 0) for n in nums) / (20 * 6)
        if spc:
            extra[9] = freq.get(spc, 0) / 20.0
    
    # 11. 连号检测
    sorted_nums = sorted(nums)
    consec = 0
    for i in range(len(sorted_nums)-1):
        if sorted_nums[i+1] - sorted_nums[i] == 1:
            consec += 1
    extra[10] = consec / 5.0
    
    # 12. 尾数分布（0-9尾）
    tails = [n % 10 for n in nums]
    extra[11] = len(set(tails)) / 6.0
    
    # 13-14. 红波蓝波绿波
    red_wave = {1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46}
    blue_wave = {3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48}
    extra[12] = sum(1 for n in nums if n in red_wave) / 6.0
    extra[13] = sum(1 for n in nums if n in blue_wave) / 6.0
    
    # 15-16. 上期重复号码
    if prev_nums_list and len(prev_nums_list) >= 1:
        prev = prev_nums_list[-1][0]
        repeat = len(set(nums) & set(prev))
        extra[14] = repeat / 6.0
        if spc:
            extra[15] = 1.0 if spc in prev else 0.0
    
    return vec, extra

# ====== 生成增强样本 ======
print("生成增强数据...")
sys.stdout.flush()

all_samples_base = []   # (x_base, y_base) for standard model
all_samples_extra = []  # (x_extra, y_extra) for enhanced model

# 基础滑动窗口
for i in range(SEQ_LEN, len(rows)):
    ctx_nums = [rows[j][0] for j in range(i-SEQ_LEN, i)]
    ctx_spc = [rows[j][1] for j in range(i-SEQ_LEN, i)]
    x_base = np.array([encode_draw_v2(n, s, rows[:i])[0] for n, s in zip(ctx_nums, ctx_spc)])
    x_extra = np.array([encode_draw_v2(n, s, rows[:i])[1] for n, s in zip(ctx_nums, ctx_spc)])
    y_base = encode_draw_v2(rows[i][0], rows[i][1], rows[:i])[0]
    y_extra = encode_draw_v2(rows[i][0], rows[i][1], rows[:i])[1]
    all_samples_base.append((x_base, y_base))
    all_samples_extra.append((x_extra, y_extra))

print(f"  基础样本: {len(all_samples_base)}")
sys.stdout.flush()

# 跳步增强
for skip in [1, 2, 3, 4, 5]:
    for i in range(SEQ_LEN, len(rows) - skip):
        if random.random() > 0.25: continue
        ctx_nums = [rows[j][0] for j in range(i-SEQ_LEN, i)]
        ctx_spc = [rows[j][1] for j in range(i-SEQ_LEN, i)]
        x_base = np.array([encode_draw_v2(n, s, rows[:i])[0] for n, s in zip(ctx_nums, ctx_spc)])
        x_extra = np.array([encode_draw_v2(n, s, rows[:i])[1] for n, s in zip(ctx_nums, ctx_spc)])
        y_base = encode_draw_v2(rows[i+skip][0], rows[i+skip][1], rows[:i+skip])[0]
        y_extra = encode_draw_v2(rows[i+skip][0], rows[i+skip][1], rows[:i+skip])[1]
        all_samples_base.append((x_base, y_base))
        all_samples_extra.append((x_extra, y_extra))

print(f"  跳步增强后: {len(all_samples_base)}")
sys.stdout.flush()

# 噪声注入（仅基础样本）
for i in range(SEQ_LEN, len(rows)-1):
    if random.random() < 0.06:
        ctx_nums = [rows[j][0] for j in range(i-SEQ_LEN, i)]
        ctx_spc = [rows[j][1] for j in range(i-SEQ_LEN, i)]
        x_base = np.array([encode_draw_v2(n, s, rows[:i])[0] for n, s in zip(ctx_nums, ctx_spc)])
        x_base += np.random.randn(*x_base.shape) * 0.02
        x_base = np.clip(x_base, 0, 1)
        y_base = encode_draw_v2(rows[i][0], rows[i][1], rows[:i])[0]
        all_samples_base.append((x_base, y_base))
        x_extra = np.array([encode_draw_v2(n, s, rows[:i])[1] for n, s in zip(ctx_nums, ctx_spc)])
        y_extra = encode_draw_v2(rows[i][0], rows[i][1], rows[:i])[1]
        all_samples_extra.append((x_extra, y_extra))

random.shuffle(all_samples_base)
random.shuffle(all_samples_extra)

# 限制样本数
MAX_SAMPLES = 200000
if len(all_samples_base) > MAX_SAMPLES:
    all_samples_base = all_samples_base[:MAX_SAMPLES]
    all_samples_extra = all_samples_extra[:MAX_SAMPLES]

print(f"  最终样本: {len(all_samples_base)}")
sys.stdout.flush()

# 分割
split = int(len(all_samples_base) * 0.8)

train_base = all_samples_base[:split]
val_base = all_samples_base[split:]
train_extra = all_samples_extra[:split]
val_extra = all_samples_extra[split:]

print(f"训练: {len(train_base)}, 验证: {len(val_base)}")
sys.stdout.flush()

# ====== Dataset ======
class LotteryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)

train_loader_base = DataLoader(
    LotteryDataset(train_base), batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=NUM_WORKERS)
val_loader_base = DataLoader(
    LotteryDataset(val_base), batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS)

train_loader_extra = DataLoader(
    LotteryDataset(train_extra), batch_size=BATCH_SIZE,
    shuffle=True, num_workers=NUM_WORKERS)
val_loader_extra = DataLoader(
    LotteryDataset(val_extra), batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS)

# ====== 模型定义 ======
# 模型A: 深Transformer（8层, 增强特征输入）
class ModelA_DeepTransformer(nn.Module):
    def __init__(self, input_dim=343, extra_dim=32, d_model=384, nhead=8, num_layers=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim + extra_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1536,
            dropout=0.15, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.global_fc = nn.Sequential(
            nn.Linear(d_model * SEQ_LEN, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, d_model))
        self.out_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(128, 49)) for _ in range(7)])
        self.dropout = nn.Dropout(0.1)
    def forward(self, x_base, x_extra):
        x = torch.cat([x_base, x_extra], dim=-1)
        B = x.shape[0]
        x = self.input_proj(x) + self.pos_encoder
        x = self.transformer(x)
        gf = self.global_fc(x.reshape(B, -1))
        last = x[:, -1, :] + gf * 0.3
        last = self.dropout(last)
        return torch.stack([h(last) for h in self.out_heads], dim=1)

# 模型B: 宽Transformer（6层, 宽FFN, 纯基础输入）
class ModelB_WideTransformer(nn.Module):
    def __init__(self, input_dim=343, d_model=512, nhead=8, num_layers=6, ffn_dim=3072):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=0.2, activation='relu', batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.Linear(d_model * SEQ_LEN, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(512, 7 * 49))
    def forward(self, x):
        B = x.shape[0]
        x = self.input_proj(x) + self.pos_encoder
        x = self.transformer(x)
        x = self.out(x.reshape(B, -1))
        return x.view(B, 7, 49)

# 模型C: LSTM变体（序列建模不同思路）
class ModelC_LSTM(nn.Module):
    def __init__(self, input_dim=343, hidden_dim=512, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2, bidirectional=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 7 * 49))
    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.out(x[:, -1, :])
        return x.view(-1, 7, 49)

# 模型D: 轻量CNN + Attention混合
class ModelD_CNN_Attention(nn.Module):
    def __init__(self, input_dim=343, d_model=256, nhead=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Sequential(
            nn.Linear(d_model * 4, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 7 * 49))
    def forward(self, x):
        B, S, D_in = x.shape
        x = self.input_proj(x)  # B, S, d_model
        x_t = x.transpose(1, 2)  # B, d_model, S
        c1 = self.conv1(x_t).transpose(1, 2)  # B, S, d_model
        c2 = self.conv2(x_t).transpose(1, 2)
        c3 = self.conv3(x_t).transpose(1, 2)
        x_attn, _ = self.attn(x, x, x)
        x_attn = self.norm(x + x_attn)
        # 每个卷积输出 d_model 维, x_attn 也是 d_model 维
        # 拼接后 = d_model * 4
        feat = torch.cat([c1, c2, c3, x_attn], dim=-1)  # B, S, d_model*4
        feat = feat[:, -1, :]  # B, d_model*4
        return self.out(feat).view(-1, 7, 49)

# ====== 训练函数 ======
def train_model(model, train_loader, val_loader, model_name, use_extra=False):
    print(f"\n{'='*50}")
    print(f"训练: {model_name}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    sys.stdout.flush()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    best_val_loss = float('inf')
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            if use_extra:
                batch_x_base, batch_x_extra, batch_y = batch[0][0], batch[0][1], batch[1]
                # 调整：extra模型使用不同数据
            batch_x, batch_y = batch
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            if use_extra and hasattr(model, 'forward') and model.__class__.__name__ == 'ModelA_DeepTransformer':
                # 需要分割base和extra
                outputs = model(batch_x[:, :, :343], batch_x[:, :, 343:])
            else:
                outputs = model(batch_x)
            loss = 0
            for pos in range(7):
                loss += loss_fn(outputs[:, pos, :], batch_y[:, pos*49:(pos+1)*49])
            loss /= 7
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                if use_extra and hasattr(model, 'forward') and model.__class__.__name__ == 'ModelA_DeepTransformer':
                    outputs = model(batch_x[:, :, :343], batch_x[:, :, 343:])
                else:
                    outputs = model(batch_x)
                loss = 0
                for pos in range(7):
                    loss += loss_fn(outputs[:, pos, :], batch_y[:, pos*49:(pos+1)*49])
                    pred = outputs[:, pos, :].argmax(dim=1)
                    true = batch_y[:, pos*49:(pos+1)*49].argmax(dim=1)
                    val_correct += (pred == true).sum().item()
                    val_total += true.size(0)
                val_loss += loss.item() / 7
        
        avg_val = val_loss / len(val_loader)
        acc = val_correct / val_total * 100
        
        scheduler.step()
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_acc"].append(acc)
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/{model_name}_best.pth")
            no_improve = 0
        else:
            no_improve += 1
        
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | train: {avg_train:.4f} | val: {avg_val:.4f} | acc: {acc:.2f}% | best: {best_val_loss:.4f}")
        sys.stdout.flush()
        
        if no_improve >= 10:
            print(f"  🛑 早停 (10 epoch无改善)")
            break
    
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/{model_name}_final.pth")
    
    # 保存历史
    history["best_val_loss"] = best_val_loss
    history["params"] = sum(p.numel() for p in model.parameters())
    with open(f"{OUTPUT_DIR}/{model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"  ✅ {model_name} 训练完成! best_val_loss={best_val_loss:.4f}")
    sys.stdout.flush()
    return best_val_loss

# ====== 并行训练 ======
print("\n\n🚀 开始多模型并行训练...")
sys.stdout.flush()

models_config = [
    (ModelA_DeepTransformer(343, 32), train_loader_base, "model_a_deep"),
    (ModelB_WideTransformer(343, 512, 8, 6, 3072), train_loader_base, "model_b_wide"),
    (ModelC_LSTM(343, 512, 4), train_loader_base, "model_c_lstm"),
    (ModelD_CNN_Attention(343, 256, 4), train_loader_base, "model_d_cnn"),
]

# 改用基础数据（所有模型都用VEC_DIM输入）
results = {}

# 创建额外的extra数据集（用于model A）
train_extra_combined = []
val_extra_combined = []
for (xb, yb), (xe, ye) in zip(train_base, train_extra):
    train_extra_combined.append((np.concatenate([xb, xe], axis=-1), yb))
for (xb, yb), (xe, ye) in zip(val_base, val_extra):
    val_extra_combined.append((np.concatenate([xb, xe], axis=-1), yb))

train_loader_extra_combined = DataLoader(
    LotteryDataset(train_extra_combined), batch_size=BATCH_SIZE,
    shuffle=True, num_workers=NUM_WORKERS)
val_loader_extra_combined = DataLoader(
    LotteryDataset(val_extra_combined), batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS)

# 串行训练（避免OOM，且便于观察）
for model_cls, loader, name in models_config:
    m = model_cls
    # 移除model A的特殊处理
    if name == "model_a_deep":
        # 重新定义为不依赖extra的版本
        class ModelA_Base(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(343, 384)
                self.pos_encoder = nn.Parameter(torch.randn(1, SEQ_LEN, 384) * 0.02)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=384, nhead=8, dim_feedforward=1536,
                    dropout=0.15, activation='gelu', batch_first=True)
                self.transformer = nn.TransformerEncoder(enc_layer, num_layers=8)
                self.global_fc = nn.Sequential(
                    nn.Linear(384 * SEQ_LEN, 512), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(512, 384))
                self.out_heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(384, 256), nn.GELU(), nn.Dropout(0.15),
                        nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(128, 49)) for _ in range(7)])
                self.dropout = nn.Dropout(0.1)
            def forward(self, x):
                B = x.shape[0]
                x = self.input_proj(x) + self.pos_encoder
                x = self.transformer(x)
                gf = self.global_fc(x.reshape(B, -1))
                last = x[:, -1, :] + gf * 0.3
                last = self.dropout(last)
                return torch.stack([h(last) for h in self.out_heads], dim=1)
        m = ModelA_Base()
    
    val = train_model(m, loader, val_loader_base, name)
    results[name] = val

# ====== 集成测试 ======
print(f"\n\n{'='*50}")
print("📊 集成模型测试")
sys.stdout.flush()

# 加载所有最佳模型
models_dict = {}
for name, _, _ in models_config:
    try:
        if name == "model_a_deep":
            m = ModelA_Base()
        elif name == "model_b_wide":
            m = ModelB_WideTransformer()
        elif name == "model_c_lstm":
            m = ModelC_LSTM()
        elif name == "model_d_cnn":
            m = ModelD_CNN_Attention()
        m.load_state_dict(torch.load(f"{OUTPUT_DIR}/{name}_best.pth", map_location=device))
        m.to(device).eval()
        models_dict[name] = m
        print(f"  ✅ {name} 加载成功")
    except Exception as e:
        print(f"  ❌ {name} 加载失败: {e}")

# 集成预测
def ensemble_predict(models, ctx_nums, ctx_spc):
    """多模型集成投票"""
    x = np.array([encode_draw_v2(n, s)[0] for n, s in zip(ctx_nums, ctx_spc)])
    x_t = torch.FloatTensor(x).unsqueeze(0).to(device)
    
    all_probs = []
    for name, m in models.items():
        with torch.no_grad():
            out = m(x_t)
            probs = torch.softmax(out, dim=-1).squeeze(0).cpu().numpy()
            all_probs.append(probs)
    
    # 平均概率
    avg_probs = np.mean(all_probs, axis=0)  # [7, 49]
    
    predictions = []
    for pos in range(7):
        prob = avg_probs[pos]
        top5_idx = np.argsort(prob)[-5:][::-1]
        predictions.append({
            "position": pos + 1,
            "top5": [int(idx) + 1 for idx in top5_idx],
            "top1": int(top5_idx[0]) + 1,
            "confidence": round(float(prob[top5_idx[0]]), 4),
        })
    return predictions

if models_dict:
    recent = rows[-SEQ_LEN:]
    ctx_nums = [r[0] for r in recent]
    ctx_spc = [r[1] for r in recent]
    preds = ensemble_predict(models_dict, ctx_nums, ctx_spc)
    
    print(f"\n🎯 集成预测结果:")
    for p in preds:
        star = "⭐" if p["position"] == 1 else ""
        print(f"  {star} 位置{p['position']}: Top1={p['top1']:2d} | Top5={p['top5']} | 置信度={p['confidence']:.4f}")
    
    # 保存集成模型配置
    ensemble_config = {
        "models": list(models_dict.keys()),
        "results": {n: float(r) for n, r in results.items()},
        "current_predictions": preds,
    }
    with open(f"{OUTPUT_DIR}/ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)

print(f"\n🎉 多模型集成训练完成！")
print(f"  模型保存至: {OUTPUT_DIR}/")
print(f"  各模型最佳val_loss: {results}")
sys.stdout.flush()
