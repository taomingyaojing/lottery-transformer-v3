#!/usr/bin/env python3
"""
构造 Qwen LoRA 微调训练数据 (llama.cpp finetune 格式)
输出: 一篇长文本，每行一个"预测"样本
"""
import json, os, random

DATA_PATH = "/home/ubuntu/lotto_raw.json"
OUTPUT_DIR = "/home/ubuntu/qwen_ft_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)

with open(DATA_PATH) as f:
    data = json.load(f)

print(f"总期数: {len(data)}")

# 用多种上下文长度构造样本
def make_samples(data, context_lens=[5, 8, 10, 15], max_samples=3000):
    samples = []
    
    for ctx_len in context_lens:
        for i in range(ctx_len, len(data)):
            target = data[i]
            ctx = data[i-ctx_len:i]
            
            # 构建文本 (用Qwen chat格式)
            prompt = f"根据近{ctx_len}期彩票开奖号码预测下一期。\n近{ctx_len}期号码：\n"
            for j, d in enumerate(ctx):
                nums_str = ",".join(f"{n:02d}" for n in d['nums'][:6])
                prompt += f"第{j+1}期 ({d['date']}): 平码:{nums_str} 特码:{d['nums'][6]:02d}\n"
            
            nums = target['nums']
            answer = json.dumps({"nums": nums[:6], "spc": nums[6]}, ensure_ascii=False)
            
            # Qwen chat template格式
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n"
            
            samples.append({
                "text": text,
                "period": target["period"],
                "date": target["date"]
            })
    
    # 限制样本量，但保持多样性
    if len(samples) > max_samples:
        random.shuffle(samples)
        samples = sorted(samples[:max_samples], key=lambda x: x["period"])
    
    return samples

samples = make_samples(data, context_lens=[5, 8, 10], max_samples=2000)
print(f"生成 {len(samples)} 个样本")

# 按时间分割 (前80%训练, 后20%验证)
split_idx = int(len(samples) * 0.8)
train_samples = samples[:split_idx]
val_samples = samples[split_idx:]

# 写到文本文件 (llama.cpp finetune --file 需要纯文本)
train_text = "".join(s["text"] for s in train_samples)
val_text = "".join(s["text"] for s in val_samples)

with open(f"{OUTPUT_DIR}/train.txt", "w") as f:
    f.write(train_text)

with open(f"{OUTPUT_DIR}/val.txt", "w") as f:
    f.write(val_text)

print(f"训练集: {len(train_samples)} 样本 -> {len(train_text)} tokens (approx)")
print(f"验证集: {len(val_samples)} 样本 -> {len(val_text)} tokens (approx)")

# 统计
total_tokens_approx = len(train_text.split()) + len(val_text.split())
print(f"总token数(approx): {total_tokens_approx}")
print(f"\n样本示例:")
print(train_samples[0]["text"][:300])
