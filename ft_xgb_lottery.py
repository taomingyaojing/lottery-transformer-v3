#!/usr/bin/env python3
"""
XGBoost 彩票预测 — 与 V3 Transformer 互补
特征工程: 序列统计特征 + 冷热度 + gap + 周期模式
训练: 7个二分类器 (6平码+1特码) + 2个全盘分类器
"""
import csv, json, os, sys, random, math
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter, defaultdict
from datetime import datetime

# ===== 配置 =====
DATA_PATH = "/home/ubuntu/lottery_bert_research/data/lottery_all_years_updated_20260423.csv"
OUTPUT_DIR = "/home/ubuntu/lottery_bert_research/ft_xgb_model"
PRED_DIR = "/home/ubuntu/lottery_bert_research/daily_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

device = "cpu"
print(f"设备: {device}")

# ===== 读取数据 =====
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

print(f"总期数: {len(rows)}")
total = len(rows)

# ===== 特征工程 =====
def extract_features(rows, idx):
    """
    对第idx期提取特征 (使用idx之前的期数做统计, 不偷窥未来)
    """
    if idx < 1:
        return None
    
    past = rows[:idx]  # 只看历史
    prev = rows[idx-1] if idx >= 1 else None
    
    nums_prev = prev[0] if prev else []
    spc_prev = prev[1] if prev else None
    
    feats = {}
    
    # --- 1. 全局统计特征 ---
    # 各位置的历史频率 (反映长期冷热度)
    pos_freq = np.zeros((6, 49), dtype=np.float32)
    spc_freq = np.zeros(49, dtype=np.float32)
    for pnums, pspc in past:
        for pi, n in enumerate(pnums):
            if 1 <= n <= 49: pos_freq[pi, n-1] += 1
        if pspc and 1 <= pspc <= 49: spc_freq[pspc-1] += 1
    
    total_past = len(past)
    pos_freq_norm = pos_freq / max(total_past, 1)
    spc_freq_norm = spc_freq / max(total_past, 1)
    
    for pi in range(6):
        feats[f'pos{pi+1}_freq'] = pos_freq_norm[pi]
    feats['spc_freq'] = spc_freq_norm
    
    # 全局出现总频率 (不分位置)
    global_freq = np.zeros(49, dtype=np.float32)
    for pnums, pspc in past:
        for n in pnums:
            if 1 <= n <= 49: global_freq[n-1] += 1
        if pspc and 1 <= pspc <= 49: global_freq[pspc-1] += 1
    global_freq_norm = global_freq / max(total_past, 1)
    feats['global_freq'] = global_freq_norm
    
    # --- 2. 近期频率 (最近10/20/30/50期) ---
    for window, label in [(10, '10'), (20, '20'), (30, '30'), (50, '50')]:
        recent = past[-min(window, len(past)):]
        rfreq = np.zeros(49, dtype=np.float32)
        for pnums, pspc in recent:
            for n in pnums:
                if 1 <= n <= 49: rfreq[n-1] += 1
            if pspc and 1 <= pspc <= 49: rfreq[pspc-1] += 1
        rfreq = rfreq / max(len(recent) * 7, 1)
        feats[f'recent_{label}_freq'] = rfreq
    
    # --- 3. Gap特征 (距离上次出现) ---
    gaps = np.full(49, len(past), dtype=np.int32)
    for offset in range(len(past)):
        pnums, pspc = past[len(past)-1-offset]
        for n in list(pnums) + ([pspc] if pspc else []):
            if 1 <= n <= 49 and gaps[n-1] == len(past):
                gaps[n-1] = offset
    feats['gap'] = gaps.astype(np.float32) / max(len(past), 1)
    
    # gap分类特征
    gap_cat = np.zeros(49, dtype=np.int32)
    for n in range(49):
        if gaps[n] <= 3: gap_cat[n] = 0  # 热号
        elif gaps[n] <= 10: gap_cat[n] = 1  # 温号
        elif gaps[n] <= 30: gap_cat[n] = 2  # 冷号
        else: gap_cat[n] = 3  # 极冷
    feats['gap_cat'] = gap_cat.astype(np.float32)
    
    # --- 4. 上一期特征 ---
    if prev:
        for pi, n in enumerate(prev[0]):
            feats[f'prev_pos{pi+1}'] = float(n) / 49.0
        feats['prev_spc'] = float(prev[1]) / 49.0 if prev[1] else 0.0
    
    # --- 5. 序列模式：相邻期增量 ---
    if len(past) >= 2:
        last2 = past[-2:]
        for pi in range(6):
            diff = last2[1][0][pi] - last2[0][0][pi]
            feats[f'delta_pos{pi+1}'] = diff / 49.0
        if last2[1][1] and last2[0][1]:
            feats['delta_spc'] = (last2[1][1] - last2[0][1]) / 49.0
    
    # --- 6. 奇偶比统计 ---
    odd_ratio = []
    for pnums, _ in past[-30:]:
        odd_count = sum(1 for n in pnums if n % 2 == 1)
        odd_ratio.append(odd_count / 6.0)
    feats['odds_ratio_mean'] = np.mean(odd_ratio) if odd_ratio else 0.5
    feats['odds_ratio_std'] = np.std(odd_ratio) if odd_ratio else 0.0
    
    # --- 7. 和值特征 ---
    sums = [sum(pnums) for pnums, _ in past[-30:]]
    feats['sum_mean'] = np.mean(sums) / 300.0 if sums else 0.5
    feats['sum_std'] = np.std(sums) / 300.0 if sums else 0.0
    
    # --- 8. 跨度特征 (max-min) ---
    spans = [(max(pnums) - min(pnums)) / 49.0 for pnums, _ in past[-30:]]
    feats['span_mean'] = np.mean(spans) if spans else 0.5
    feats['span_std'] = np.std(spans) if spans else 0.0
    
    # --- 9. 上期号码的邻号统计 ---
    # 每个号码±1,±2的号码在近期的出现频率
    neighbor_recent = np.zeros(49, dtype=np.float32)
    recent_nums = set()
    for pnums, pspc in past[-20:]:
        for n in pnums:
            recent_nums.add(n)
        if pspc:
            recent_nums.add(pspc)
    
    total_recent_nums = len(recent_nums)
    feats['recent_unique'] = total_recent_nums / 49.0
    
    return feats


def flatten_feats(feats_dict, prefix=""):
    """将特征字典展平为一维数组"""
    vec = []
    for key, val in feats_dict.items():
        if isinstance(val, np.ndarray):
            vec.extend(val.flatten())
        elif isinstance(val, (int, float, np.integer, np.floating)):
            vec.append(float(val))
        else:
            pass
    return np.array(vec, dtype=np.float32)


def get_feat_names(feats_dict, prefix=""):
    """获取特征名称列表"""
    names = []
    for key, val in feats_dict.items():
        if isinstance(val, np.ndarray):
            for i in range(val.size):
                names.append(f"{prefix}{key}_{i}")
        elif isinstance(val, (int, float, np.integer, np.floating)):
            names.append(f"{prefix}{key}")
    return names


# ===== 生成训练数据 =====
print("\n生成训练特征...")
sys.stdout.flush()

all_features = []
all_targets_pos = []
all_targets_spc = []

# 每个样本预测第idx期
MIN_HISTORY = 60
for idx in range(MIN_HISTORY, len(rows)):
    feats = extract_features(rows, idx)
    if feats is None:
        continue
    
    # 展平特征
    fvec = flatten_feats(feats)
    all_features.append(fvec)
    
    # 目标：7个位置的真实号码 (1-49 → 0-48)
    target_nums = rows[idx][0]  # [6个平码]
    target_spc = rows[idx][1]   # 特码
    all_targets_pos.append([n-1 for n in target_nums])
    all_targets_spc.append(target_spc - 1 if target_spc else 0)

print(f"总样本数: {len(all_features)}")
print(f"特征维度: {len(all_features[0])}")

# 特征名称
sample_feats = extract_features(rows, MIN_HISTORY)
feat_names = get_feat_names(sample_feats)
print(f"特征名称 ({len(feat_names)}): {feat_names[:15]}...")

# ===== 划分训练/验证集 =====
# 按时间序，前80%训练，后20%验证
split_idx = int(len(all_features) * 0.8)
train_X = np.array(all_features[:split_idx])
train_Y_pos = np.array(all_targets_pos[:split_idx])
train_Y_spc = np.array(all_targets_spc[:split_idx])

val_X = np.array(all_features[split_idx:])
val_Y_pos = np.array(all_targets_pos[split_idx:])
val_Y_spc = np.array(all_targets_spc[split_idx:])

print(f"\n训练: {len(train_X)}, 验证: {len(val_X)}")
sys.stdout.flush()

# ===== 训练: 每个位置一个多分类器 =====
# 也训练联合分类器
print("\n" + "=" * 60)
print("🦦 训练 XGBoost 模型群")
print("=" * 60)
sys.stdout.flush()

models_pos = {}    # 6个平码分类器
model_spc = None   # 特码分类器
model_all = None   # 全盘多分类
model_direct = None  # 直接预测6个不重复号码

params = {
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'eval_metric': 'mlogloss',
    'objective': 'multi:softprob',
    'num_class': 49,
    'random_state': 42,
    'n_jobs': 4,
    'verbosity': 0,
}

# 训练7个独立位置分类器
for pi in range(6):
    print(f"\n训练位置{pi+1}分类器...")
    sys.stdout.flush()
    
    dtrain = xgb.DMatrix(train_X, label=train_Y_pos[:, pi], feature_names=feat_names)
    dval = xgb.DMatrix(val_X, label=val_Y_pos[:, pi], feature_names=feat_names)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=100
    )
    
    # 验证准确率
    val_probs = model.predict(dval)
    val_pred = val_probs.argmax(axis=1)
    acc = (val_pred == val_Y_pos[:, pi]).mean()
    print(f"  位置{pi+1} 验证准确率: {acc*100:.2f}%")
    sys.stdout.flush()
    
    models_pos[pi] = model
    model.save_model(f"{OUTPUT_DIR}/xgb_pos{pi+1}.json")

print("\n训练特码分类器...")
sys.stdout.flush()

dtrain_spc = xgb.DMatrix(train_X, label=train_Y_spc, feature_names=feat_names)
dval_spc = xgb.DMatrix(val_X, label=val_Y_spc, feature_names=feat_names)

model_spc = xgb.train(
    params,
    dtrain_spc,
    num_boost_round=500,
    evals=[(dtrain_spc, 'train'), (dval_spc, 'val')],
    early_stopping_rounds=30,
    verbose_eval=100
)

val_probs_spc = model_spc.predict(dval_spc)
val_pred_spc = val_probs_spc.argmax(axis=1)
acc_spc = (val_pred_spc == val_Y_spc).mean()
print(f"  特码验证准确率: {acc_spc*100:.2f}%")
sys.stdout.flush()

model_spc.save_model(f"{OUTPUT_DIR}/xgb_spc.json")

print("\n✅ XGBoost 模型训练完成！")
sys.stdout.flush()

# ===== 回测 =====
print("\n" + "=" * 60)
print("📊 滚动回测 (最近200期)")
print("=" * 60)
sys.stdout.flush()

correct_top1 = [0]*7
correct_top3 = [0]*7
correct_top5 = [0]*7
total_test = 0

test_start = max(MIN_HISTORY, len(rows) - 200)

for idx in range(test_start, len(rows)):
    feats = extract_features(rows, idx)
    if feats is None: continue
    
    x = flatten_feats(feats).reshape(1, -1)
    dtest = xgb.DMatrix(x, feature_names=feat_names)
    
    true_nums = rows[idx][0]
    true_spc = rows[idx][1]
    
    preds = []
    for pi in range(6):
        probs = models_pos[pi].predict(dtest)[0]
        preds.append(probs)
    spc_probs = model_spc.predict(dtest)[0]
    
    for pi in range(6):
        pred_num = preds[pi].argmax()
        true_num = true_nums[pi] - 1
        top3 = np.argsort(preds[pi])[-3:]
        top5 = np.argsort(preds[pi])[-5:]
        
        if pred_num == true_num: correct_top1[pi] += 1
        if true_num in top3: correct_top3[pi] += 1
        if true_num in top5: correct_top5[pi] += 1
    
    pred_spc = spc_probs.argmax()
    true_spc_idx = true_spc - 1 if true_spc else 0
    spc_top3 = np.argsort(spc_probs)[-3:]
    spc_top5 = np.argsort(spc_probs)[-5:]
    
    if pred_spc == true_spc_idx: correct_top1[6] += 1
    if true_spc_idx in spc_top3: correct_top3[6] += 1
    if true_spc_idx in spc_top5: correct_top5[6] += 1
    
    total_test += 1

print(f"\n回测 {total_test} 期:")
for p in range(7):
    label = f"位置{p+1}" if p < 6 else "特码"
    t1 = correct_top1[p]
    t3 = correct_top3[p]
    t5 = correct_top5[p]
    print(f"  {label}: top1={t1}({t1/total_test*100:.1f}%) top3={t3}({t3/total_test*100:.1f}%) top5={t5}({t5/total_test*100:.1f}%)")
sys.stdout.flush()

# ===== 最终预测 =====
print("\n" + "=" * 60)
print("🔮 XGBoost 最终预测")
print("=" * 60)
sys.stdout.flush()

# 多窗口预测 (3/5/8/10期)
window_sizes = [3, 5, 8, 10, 15]
votes_pos = np.zeros((7, 49))

for ws in window_sizes:
    if ws > len(rows):
        continue
    
    # 伪索引：用最后ws期数据生成特征
    # 对最后ws期分别提取特征，平均预测
    for step in range(ws):
        ref_idx = len(rows) - ws + step
        if ref_idx < MIN_HISTORY:
            continue
        
        feats = extract_features(rows, ref_idx)
        if feats is None: continue
        
        x = flatten_feats(feats).reshape(1, -1)
        dtest = xgb.DMatrix(x, feature_names=feat_names)
        
        for pi in range(6):
            probs = models_pos[pi].predict(dtest)[0]
            votes_pos[pi] += probs
        
        spc_probs = model_spc.predict(dtest)[0]
        votes_pos[6] += spc_probs

# 平均
votes_pos /= (len(window_sizes) * len(window_sizes))

final = votes_pos.argmax(axis=1)
print(f"\n平码: {[int(final[i])+1 for i in range(6)]}")
print(f"特码: {int(final[6])+1}")

print(f"\n📊 每位置 Top5:")
for p in range(7):
    label = f"位置{p+1}" if p < 6 else "特码"
    top5 = np.argsort(votes_pos[p])[-5:][::-1]
    top5_str = ", ".join([f"{int(idx)+1}({votes_pos[p][idx]:.3f})" for idx in top5])
    print(f"  {label}: {top5_str}")
sys.stdout.flush()

# ===== 保存结果 =====
backtest = {}
for p in range(7):
    label = f"pos{p+1}" if p < 6 else "spc"
    backtest[f"{label}_top1"] = f"{correct_top1[p]}/{total_test}({correct_top1[p]/total_test*100:.1f}%)"
    backtest[f"{label}_top3"] = f"{correct_top3[p]}/{total_test}({correct_top3[p]/total_test*100:.1f}%)"
    backtest[f"{label}_top5"] = f"{correct_top5[p]}/{total_test}({correct_top5[p]/total_test*100:.1f}%)"

output = {
    "engine": "XGBoost_Lottery_v1",
    "train_config": {
        "max_depth": params['max_depth'],
        "learning_rate": params['learning_rate'],
        "n_estimators": params['n_estimators'],
        "feature_dim": len(all_features[0]),
        "train_samples": len(train_X),
        "val_samples": len(val_X),
        "feature_names": feat_names,
    },
    "validation": {
        "pos_accuracies": [f"{(train_Y_pos[:100, pi] == models_pos[pi].predict(xgb.DMatrix(train_X[:100], feature_names=feat_names)).argmax(axis=1)).mean()*100:.1f}%" for pi in range(6)],
    },
    "backtest": backtest,
    "prediction": {
        "nums": [int(final[i])+1 for i in range(6)],
        "spc": int(final[6])+1,
        "position_top5": {
            f"pos{p+1}" if p < 6 else "spc": [int(np.argsort(votes_pos[p])[-5:][::-1][i])+1 for i in range(5)]
            for p in range(7)
        }
    }
}

with open(f"{OUTPUT_DIR}/xgb_result.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ 结果已保存: {OUTPUT_DIR}/xgb_result.json")
print(f"⏱️ 完成!")
sys.stdout.flush()
