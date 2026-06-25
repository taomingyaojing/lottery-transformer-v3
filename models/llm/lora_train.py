#!/usr/bin/env python3
"""
Qwen2.5-1.5B LoRA 微调 — 彩票序列预测
用历史开奖数据训练Qwen生成下期号码

方案: 4bit量化 + LoRA (rank=8) + CPU offload
"""
import json, os, sys, time, math, re
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# ===== 配置 =====
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DATA_PATH = "/home/ubuntu/lotto_raw.json"
CACHE_DIR = "/home/ubuntu/.cache/huggingface"
OUTPUT_DIR = "/home/ubuntu/qwen_lora_lottery"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LoRA参数
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

device = torch.device("cpu")
print(f"设备: {device}")
print(f"CPU 线程: {torch.get_num_threads()}")
sys.stdout.flush()

# ===== 加载数据 =====
print("\n加载彩票数据...")
with open(DATA_PATH) as f:
    data = json.load(f)
print(f"  共 {len(data)} 期")

# ===== 构造训练样本 (指令格式) =====
"""
格式：
Human: 根据近10期彩票开奖号码预测下一期。
近10期号码：
第1期: 01,05,12,23,34,45 特码:08
...
第10期: xx,xx,xx,xx,xx,xx 特码:xx

Assistant: 预测结果：
[JSON]
{"nums": [xx, xx, xx, xx, xx, xx], "spc": xx}
[/JSON]
"""

def format_nums(nums, spc):
    return f"平码:{','.join(f'{n:02d}' for n in nums)} 特码:{spc:02d}"

def make_samples(data, context_lens=[5, 8, 10, 15], max_samples=5000):
    samples = []
    
    for ctx_len in context_lens:
        for i in range(ctx_len, len(data)):
            ctx = data[i-ctx_len:i]
            target = data[i]
            
            # 构建对话
            history = []
            for j, d in enumerate(ctx):
                history.append(f"第{j+1}期 ({d['date']}): {format_nums(d['nums'][:6], d['nums'][6])}")
            
            nums_str = target['nums']
            answer = json.dumps({"nums": nums_str[:6], "spc": nums_str[6]}, ensure_ascii=False)
            
            prompt = f"根据近{ctx_len}期彩票开奖号码预测下一期。\n近{ctx_len}期号码：\n" + "\n".join(history)
            response = f"预测结果：\n[JSON]\n{answer}\n[/JSON]"
            
            samples.append({
                "prompt": prompt,
                "response": response,
                "ctx_len": ctx_len,
                "period": target["period"]
            })
    
    # 限制样本量
    if len(samples) > max_samples:
        import random
        random.shuffle(samples)
        samples = samples[:max_samples]
    
    return samples

print("\n构造训练样本...")
samples = make_samples(data, context_lens=[5, 8, 10, 15], max_samples=4000)
print(f"  共 {len(samples)} 个样本")

# 分割 (最早期的数据用于训练，最新的做验证)
mid = int(len(data) * 0.8)
# 用时间分割: 最早80%数据点对应的样本做训练
# 找到 mid 对应的 period
mid_period = data[mid]["period"]
train_samples = [s for s in samples if s["period"] <= mid_period]
val_samples = [s for s in samples if s["period"] > mid_period]
print(f"  训练: {len(train_samples)}, 验证: {len(val_samples)}")

# ===== 格式化数据 =====
def format_conversation(prompt, response):
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"

train_texts = [format_conversation(s["prompt"], s["response"]) for s in train_samples]
val_texts = [format_conversation(s["prompt"], s["response"]) for s in val_samples]

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})

# ===== 加载模型和tokenizer =====
print("\n加载Qwen2.5-1.5B (4bit量化)...")
sys.stdout.flush()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="cpu",
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
print(f"  模型加载成功!")
sys.stdout.flush()

# ===== 设置 LoRA =====
print("\n配置 LoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
sys.stdout.flush()

# ===== Tokenize =====
def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding=False,
        return_tensors=None,
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

print("\nTokenizing...")
sys.stdout.flush()
train_tokenized = train_dataset.map(tokenize_function, remove_columns=["text"], batched=True, num_proc=1)
val_tokenized = val_dataset.map(tokenize_function, remove_columns=["text"], batched=True, num_proc=1)

# ===== 训练 =====
print("\n开始 LoRA 训练...")
sys.stdout.flush()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,      # CPU 小batch
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,       # 累积模拟batch
    warmup_steps=50,
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=False,                          # CPU不支持fp16
    bf16=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    report_to="none",
    ddp_find_unused_parameters=False,
    max_grad_norm=0.3,
    save_total_limit=2,
    lr_scheduler_type="cosine",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
)

start_time = time.time()
trainer.train()
total_time = time.time() - start_time
print(f"\n训练完成! 耗时: {total_time/60:.1f} 分钟")
sys.stdout.flush()

# ===== 保存 LoRA =====
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
print(f"\nLoRA适配器已保存到: {OUTPUT_DIR}/lora_adapter")
sys.stdout.flush()

# ===== 预测测试 =====
print("\n=== 预测测试 ===")
print("加载 LoRA 进行推理...")
sys.stdout.flush()

# 用最新几期做预测
test_ctx = 10
ctx_data = data[-test_ctx-5:-5]  # 用中间一段做验证
target_data = data[-5:]          # 最后5期做测试

correct_top3_nums = 0
correct_spc = 0
total_test = 0

# 合并LoRA到模型进行推理
model.eval()

for td in target_data:
    ctx = data[data.index(td)-test_ctx:data.index(td)]
    
    history = []
    for j, d in enumerate(ctx):
        history.append(f"第{j+1}期 ({d['date']}): {format_nums(d['nums'][:6], d['nums'][6])}")
    
    prompt = f"根据近{test_ctx}期彩票开奖号码预测下一期。\n近{test_ctx}期号码：\n" + "\n".join(history)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # 提取预测号码
    nums_found = [int(x) for x in re.findall(r'\b([1-9]|[1-4][0-9])\b', response)]
    nums_found = [n for n in nums_found if 1 <= n <= 49]
    
    true_nums = td["nums"][:6]
    true_spc = td["nums"][6]
    
    print(f"\n  实际: {true_nums} + T{true_spc}")
    print(f"  预测: {nums_found[:7]}")
    
    if len(nums_found) >= 6:
        match = sum(1 for n in nums_found[:6] if n in true_nums)
        print(f"  平码命中: {match}/6")
        correct_top3_nums += match
    
    if len(nums_found) >= 7 and nums_found[6] == true_spc:
        correct_spc += 1
        print(f"  特码命中! ✅")
    elif len(nums_found) >= 7:
        print(f"  特码: 预测{nums_found[6]}, 实际{true_spc} ❌")
    
    total_test += 1

print(f"\n=== 测试结果 ({total_test}期) ===")
print(f"  平码平均命中: {correct_top3_nums/total_test:.1f}/6")
print(f"  特码命中率: {correct_spc}/{total_test} ({correct_spc/total_test*100:.0f}%)")
sys.stdout.flush()

print("\n🦦 Qwen LoRA 训练完成!")
