#!/usr/bin/env python3
"""
📊 噪声实验：把上期某个号码改一点点，看预测变不变
"""
import csv, json, os, sys, re, time, subprocess, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb

DATA_PATH = "/home/ubuntu/lottery_bert_research/data/lottery_all_years_updated_20260423.csv"
V3_MODEL_DIR = "/home/ubuntu/lottery_bert_research/ft_model_v3"
XGB_MODEL_DIR = "/home/ubuntu/lottery_bert_research/ft_xgb_model"
LLM_MODEL_PATH = os.path.expanduser("~/qwen2.5-1.5b-q4.gguf")
LLAMA_SERVER_PATH = os.path.expanduser("~/llama.cpp/build/bin/llama-server")

SEQ_LEN = 8; VEC_DIM = 343; LLM_PORT = 8082

def load_data():
    rows = []
    with open(DATA_PATH) as f:
        for r in csv.reader(f):
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
    return rows

rows = load_data()
print(f"原始数据: {len(rows)}期")
print(f"上期: {rows[-1][0]} + T{rows[-1][1]}")

# 复制数据，把上期第一个号码28改成28.5
import copy
rows_noise = copy.deepcopy(rows)
last = rows_noise[-1]
# 找28并改成28.5
last_nums = last[0]
if 28 in last_nums:
    idx_28 = last_nums.index(28)
    # 改成28.5 — 本质是在编码时把28这个one-hot变成0.5
    # 但one-hot只接受int，所以这个没办法正常编码
    print(f"找到28在位置{idx_28+1}, 改为28.5")

# 实际上one-hot没法编码28.5，所以换个思路：
# 不改数字，只改编码——在V3的one-hot编码里把28的位置设为0.5而不是1.0
# 对XGBoost：把28改成29来模拟"微调"（最接近的变化）

# === 实验设计 ===
# 实验A: V3 — 把上期28在编码时设为0.5而不是1.0
# 实验B: XGBoost — 把上期28改成29（相邻整数，模拟微调）
# 实验C: LLM — 把上期28改成29

print("\n" + "=" * 70)
print("📊 实验A: V3 — 上期28编码强度从1.0降到0.5")
print("=" * 70)

class LotteryTransformerV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(343, 512)
        self.pos_encoder = nn.Parameter(torch.randn(1, 8, 512) * 0.02)
        el = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(el, num_layers=6)
        self.global_fc = nn.Sequential(nn.Linear(512*8, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 512))
        self.output_heads = nn.ModuleList([nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.15), nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 49)) for _ in range(7)])
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

def encode_draw_custom(nums, spc=None, weaken_map=None):
    """自定义编码，weaken_map = {位置: 强度值}"""
    vec = np.zeros(VEC_DIM, dtype=np.float32)
    for i, n in enumerate(nums):
        if 1 <= n <= 49:
            val = 1.0
            if weaken_map and i in weaken_map:
                val = weaken_map[i]
            vec[i * 49 + (n - 1)] = val
    if spc and 1 <= spc <= 49: vec[6 * 49 + (spc - 1)] = 1.0
    return vec

def v3_predict(rows_data, weaken_map=None):
    votes = np.zeros((7, 49))
    for ctx in [3, 5, 8, 10]:
        if ctx > len(rows_data): continue
        seq_nums = [rows_data[-i][0] for i in range(ctx, 0, -1)]
        seq_spc = [rows_data[-i][1] for i in range(ctx, 0, -1)]
        x_vecs = []
        for i, (ns, sp) in enumerate(zip(seq_nums, seq_spc)):
            wm = weaken_map if i == len(seq_nums)-1 else None  # 只对最后一期减弱
            x_vecs.append(encode_draw_custom(ns, sp, wm))
        x_vecs = np.array(x_vecs)
        if ctx < 8:
            pad = np.zeros((8 - ctx, 343))
            x_vecs = np.vstack([pad, x_vecs])
        elif ctx > 8:
            x_vecs = x_vecs[-8:]
        with torch.no_grad():
            pred = v3_model(torch.FloatTensor(x_vecs).unsqueeze(0))
        votes += F.softmax(pred, dim=-1).squeeze(0).numpy()
    return votes

v3_model = LotteryTransformerV3()
v3_model.load_state_dict(torch.load(f"{V3_MODEL_DIR}/best_model.pth", map_location=device, weights_only=True))
v3_model.eval()

# 原始预测
v3_orig = v3_predict(rows, weaken_map=None)
v3_orig_pred = v3_orig.argmax(axis=1)

# 找到上期最后一个28的位置
last_nums = rows[-1][0]
weaken_positions = {}
for i, n in enumerate(last_nums):
    if n == 28:
        weaken_positions[i] = 0.5  # 强度降到0.5
        print(f"  将位置{i+1}的28编码强度降为0.5")

if weaken_positions:
    v3_noise = v3_predict(rows, weaken_map=weaken_positions)
    v3_noise_pred = v3_noise.argmax(axis=1)
    
    orig_nums = [int(v3_orig_pred[i])+1 for i in range(7)]
    noise_nums = [int(v3_noise_pred[i])+1 for i in range(7)]
    
    print(f"  原始预测: {orig_nums[:6]} + T{orig_nums[6]}")
    print(f"  减弱预测: {noise_nums[:6]} + T{noise_nums[6]}")
    changes = sum(1 for i in range(7) if orig_nums[i] != noise_nums[i])
    print(f"  变化: {changes}/7")
else:
    print("  上期没有28")

# ===== 实验B: XGBoost — 28改成29 =====
print("\n" + "=" * 70)
print("📊 实验B: XGBoost — 上期28改成29")
print("=" * 70)

xgb_ok = all(os.path.exists(f"{XGB_MODEL_DIR}/xgb_pos{pi+1}.json") for pi in range(6))
if xgb_ok:
    with open(f"{XGB_MODEL_DIR}/xgb_result.json") as f:
        feat_names = json.load(f)['train_config']['feature_names']
    
    models_pos = {}
    for pi in range(6):
        m = xgb.Booster(); m.load_model(f"{XGB_MODEL_DIR}/xgb_pos{pi+1}.json"); models_pos[pi] = m
    model_spc = xgb.Booster(); model_spc.load_model(f"{XGB_MODEL_DIR}/xgb_spc.json")
    
    def extract_features(rows, idx):
        if idx < 1: return None
        past = rows[:idx]; prev = rows[idx-1] if idx >= 1 else None
        feats = {}
        pos_freq = np.zeros((6, 49)); spc_freq = np.zeros(49)
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
        for k, val in fd.items():
            if isinstance(val, np.ndarray): v.extend(val.flatten())
            elif isinstance(val, (int, float, np.integer, np.floating)): v.append(float(val))
        return np.array(v, dtype=np.float32)
    
    def xgb_predict(rows_data):
        votes = np.zeros((7, 49))
        for w in [3, 5, 8, 10, 15]:
            for step in range(w):
                ri = len(rows_data) - w + step
                if ri < 60: continue
                f = extract_features(rows_data, ri)
                if f is None: continue
                x = flatten_feats(f).reshape(1, -1)
                d = xgb.DMatrix(x, feature_names=feat_names)
                for pi in range(6): votes[pi] += models_pos[pi].predict(d)[0]
                votes[6] += model_spc.predict(d)[0]
        votes /= 5*5
        return votes
    
    # 原始
    xgb_orig = xgb_predict(rows)
    xgb_orig_pred = xgb_orig.argmax(axis=1)
    
    # 修改：上期28改成29
    rows_mod = copy.deepcopy(rows)
    for i, n in enumerate(rows_mod[-1][0]):
        if n == 28:
            rows_mod[-1][0][i] = 29
            print(f"  将位置{i+1}从28改为29")
    
    xgb_mod = xgb_predict(rows_mod)
    xgb_mod_pred = xgb_mod.argmax(axis=1)
    
    orig_nums = [int(xgb_orig_pred[i])+1 for i in range(7)]
    mod_nums = [int(xgb_mod_pred[i])+1 for i in range(7)]
    
    print(f"  原始预测: {orig_nums[:6]} + T{orig_nums[6]}")
    print(f"  修改后:   {mod_nums[:6]} + T{mod_nums[6]}")
    changes = sum(1 for i in range(7) if orig_nums[i] != mod_nums[i])
    print(f"  变化: {changes}/7")
else:
    print("  ⏭️ 跳过")

# ===== 实验C: LLM — 28改成29 =====
print("\n" + "=" * 70)
print("📊 实验C: LLM — 上期28改成29")
print("=" * 70)

if os.path.exists(LLM_MODEL_PATH) and os.path.exists(LLAMA_SERVER_PATH):
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
    
    def llm_predict(rows_data):
        recent = rows_data[-12:]
        seq_parts = ["-".join(f"{n:02d}" for n in ns) + f" {spc:02d}" for ns, spc in recent]
        prompt = "Lottery sequence (last 12 draws):\n\n" + "\n".join(seq_parts) + "\n\nNext line:"
        
        data = json.dumps({"prompt": prompt, "n_predict": 30, "temperature": 0.5, "cache_prompt": False}).encode()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(f"http://127.0.0.1:{LLM_PORT}/completion", data=data, headers={"Content-Type": "application/json"}),
                timeout=60
            ) as resp:
                result = json.loads(resp.read()).get("content", "")
                nums = [int(x) for x in re.findall(r'\b([1-9]|[1-4][0-9])\b', result)]
                nums = [n for n in nums if 1 <= n <= 49]
                return nums if len(nums) >= 7 else None, result
        except Exception as e:
            return None, str(e)
    
    # 原始
    orig_nums, orig_raw = llm_predict(rows)
    print(f"  原始输出: {orig_raw[:80]}")
    print(f"  原始预测: {orig_nums[:6] if orig_nums else None} + T{orig_nums[-1] if orig_nums else None}")
    
    # 修改
    rows_mod_llm = copy.deepcopy(rows)
    for i, n in enumerate(rows_mod_llm[-1][0]):
        if n == 28:
            rows_mod_llm[-1][0][i] = 29
    
    mod_nums, mod_raw = llm_predict(rows_mod_llm)
    print(f"  修改输出: {mod_raw[:80]}")
    print(f"  修改预测: {mod_nums[:6] if mod_nums else None} + T{mod_nums[-1] if mod_nums else None}")
    
    if orig_nums and mod_nums:
        changes = sum(1 for i in range(min(len(orig_nums), len(mod_nums))) if orig_nums[i] != mod_nums[i])
        print(f"  变化: {changes}/{min(len(orig_nums), 7)}")
    
    llm_proc.terminate()
    try: llm_proc.wait(timeout=5)
    except: llm_proc.kill()
else:
    print("  ⏭️ 跳过")

print("\n" + "=" * 70)
print("📋 结论")
print("=" * 70)
print("""
如果模型对一个数字的微小变化(<1)有强烈反应:
  说明它在这个数字上有"高敏感性"，可能是过拟合了
如果反应很小或没有:
  说明模型鲁棒性好，大数趋势决策为主
""")
