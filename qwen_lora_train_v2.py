#!/usr/bin/env python3
"""
Qwen2.5-0.5B LoRA 微调 — 彩票序列预测 (省内存版)

省内存措施:
- 4bit量化 + NF4 / 0.5B参数量
- 使用 SGD 优化器 (省 ~3GB Adam states)
- gradient_checkpointing
- batch_size=1, grad_accum=4
- 关闭 torch.compile
- max_length=512 (缩短上下文)
"""
import json, os, sys, time, math, re, gc
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# ===== 配置 =====
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_PATH = "/home/ubuntu/lotto_raw.json"
CACHE_DIR = "/home/ubuntu/.cache/huggingface"
OUTPUT_DIR = "/home/ubuntu/qwen_lora_lottery"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # 关闭编译省内存

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

# ===== 构造样本 (简短上下文) =====
def format_nums(nums, spc):
    return f"平码:{','.join(str(n) for n in nums)} 特码:{spc}"

def make_samples(data, context_lens=[5, 8], max_samples=1500):
    samples = []
    for ctx_len in context_lens:
        for i in range(ctx_len, len(data)):
            ctx = data[i-ctx_len:i]
            target = data[i]
            
            history = []
            for j, d in enumerate(ctx):
                history.append(f"#{d['period']}:{format_nums(d['nums'][:6], d['nums'][6])}")
            
            nums = target['nums']
            answer = json.dumps({"nums": nums[:6], "spc": nums[6]}, ensure_ascii=False)
            
            prompt = f"根据近{ctx_len}期预测: {' '.join(history)}"
            response = answer
            
            samples.append({
                "prompt": prompt,
                "response": response,
                "period": target["period"],
            })
    
    if len(samples) > max_samples:
        import random
        random.shuffle(samples)
        samples = samples[:max_samples]
    return samples

print("\n构造训练样本...")
samples = make_samples(data, context_lens=[5, 8], max_samples=1500)
print(f"  共 {len(samples)} 个样本")

# 分割
mid_period = data[int(len(data)*0.8)]["period"]
train_samples = [s for s in samples if s["period"] <= mid_period]
val_samples = [s for s in samples if s["period"] > mid_period]
print(f"  训练: {len(train_samples)}, 验证: {len(val_samples)}")

def format_conversation(p, r):
    return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>"

train_texts = [format_conversation(s["prompt"], s["response"]) for s in train_samples]
val_texts = [format_conversation(s["prompt"], s["response"]) for s in val_samples]

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})

# ===== 加载模型 (省内存版) =====
print("\n加载Qwen2.5-1.5B (4bit, 省内存配置)...")
sys.stdout.flush()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=False,  # 省内存
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True
)
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
    use_cache=False,              # 省内存
)
print(f"  模型加载成功!")
sys.stdout.flush()

# ===== LoRA =====
print("\n配置 LoRA...")
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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

# ===== Tokenize (短context) =====
def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,            # 短上下文省内存
        padding=False,
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

print("\nTokenizing...")
sys.stdout.flush()
train_tokenized = train_dataset.map(tokenize_function, remove_columns=["text"], batched=True)
val_tokenized = val_dataset.map(tokenize_function, remove_columns=["text"], batched=True)

# 手动触发GC
gc.collect()

# ===== 训练 (SGD省内存) =====
print("\n开始 LoRA 训练 (SGD, batch=1, grad_accum=4)...")
sys.stdout.flush()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=30,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=30,
    save_strategy="steps",
    save_steps=60,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    learning_rate=2e-4,
    weight_decay=0.0,
    fp16=False,
    bf16=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    report_to="none",
    ddp_find_unused_parameters=False,
    max_grad_norm=1.0,
    save_total_limit=2,
    lr_scheduler_type="linear",
    optim="sgd",                     # SGD省内存!
    gradient_checkpointing=True,
    skip_memory_metrics=False,
    dataloader_num_workers=0,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
)

start_time = time.time()
gc.collect()
trainer.train()
total_time = time.time() - start_time
print(f"\n训练完成! 耗时: {total_time/60:.1f} 分钟")
sys.stdout.flush()

# ===== 保存 =====
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
print(f"\nLoRA适配器已保存到: {OUTPUT_DIR}/lora_adapter")
sys.stdout.flush()

# ===== 预测测试 =====
print("\n=== 预测测试 ===")
model.eval()

test_ctx = 8
test_data = data[-20:]

correct_top3 = 0
correct_spc = 0
total_test = 0

for td in test_data:
    idx = data.index(td)
    if idx < test_ctx: continue
    ctx = data[idx-test_ctx:idx]
    
    history = " ".join(f"#{d['period']}:{format_nums(d['nums'][:6], d['nums'][6])}" for d in ctx)
    prompt = f"根据近{test_ctx}期预测: {history}"
    
    inputs = tokenizer(
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        return_tensors="pt", truncation=True, max_length=512
    )
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=40,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    nums_found = [int(x) for x in re.findall(r'\b([1-9]|[1-4][0-9])\b', response)]
    nums_found = [n for n in nums_found if 1 <= n <= 49][:7]
    
    true_nums = td["nums"][:6]
    true_spc = td["nums"][6]
    
    print(f"  #{td['period']}: 实际 {true_nums}+T{true_spc} | 预测 {nums_found}")
    
    if len(nums_found) >= 6:
        match = sum(1 for n in nums_found[:6] if n in true_nums)
        correct_top3 += match
    if len(nums_found) >= 7 and nums_found[6] == true_spc:
        correct_spc += 1
    total_test += 1

print(f"\n=== 测试 ({total_test}期) ===")
if total_test > 0:
    print(f"  平均命中: {correct_top3/total_test:.1f}/6")
    print(f"  特码命中: {correct_spc}/{total_test} ({correct_spc/total_test*100:.0f}%)")

print("\n🦦 Qwen LoRA 训练完成!")
