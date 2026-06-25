#!/usr/bin/env python3
"""
🦦 LLM 彩票预测 — Qwen2.5-1.5B via llama.cpp
直接纯文本模式，避免chat template问题
"""
import json, csv, subprocess, os, sys, time, urllib.request, urllib.error, re

LLAMA_SERVER_PATH = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
MODEL_PATH = os.path.expanduser("~/qwen2.5-1.5b-q4.gguf")
DATA_PATH = "./data/lottery_history.csv"
PORT = 8081

server_proc = None

def load_data():
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
    return rows

def start_server():
    global server_proc
    # 先检查是否已经在运行
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
        with urllib.request.urlopen(req, timeout=2):
            return True
    except:
        pass
    
    cmd = [
        LLAMA_SERVER_PATH,
        "-m", MODEL_PATH,
        "--port", str(PORT),
        "--host", "127.0.0.1",
        "-c", "8192",
        "--no-mmap",
        "-ngl", "0",
    ]
    server_proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    for i in range(30):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except:
            time.sleep(1)
    return False

def stop_server():
    global server_proc
    if server_proc:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except:
            server_proc.kill()
        server_proc = None

def query_llm_plain(prompt, max_tokens=40, temp=0.5):
    """纯文本模式推理"""
    data = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temp,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result.get("content", "").strip()
    except Exception as e:
        return f"ERROR: {e}"

def build_sequence_prompt(rows, num_context=12):
    """构建纯数字序列prompt"""
    recent = rows[-num_context:]
    seq_parts = []
    for nums, spc in recent:
        seq_parts.append("-".join(f"{n:02d}" for n in nums) + f" {spc:02d}")
    
    seq_str = "\n".join(seq_parts)
    prompt = f"""Lottery sequence (last {num_context} draws, each line: 6nums-then-special):

{seq_str}

Next line:"""
    return prompt

def parse_numbers(text):
    """从文本中提取1-49的数字"""
    nums = [int(x) for x in re.findall(r'\b([1-9]|[1-4][0-9])\b', text)]
    return [n for n in nums if 1 <= n <= 49]

def predict_single(temp=0.5):
    """单次预测"""
    rows = load_data()
    
    prompt = build_sequence_prompt(rows, num_context=12)
    response = query_llm_plain(prompt, max_tokens=40, temp=temp)
    
    nums = parse_numbers(response)
    
    if len(nums) >= 7:
        return nums[:6], nums[-1], response
    return None, None, response

def predict_ensemble(num_runs=5):
    """多次预测取综合"""
    rows = load_data()
    
    print(f"  🧠 进行 {num_runs} 次推理取综合...")
    
    all_votes = {}
    
    for run in range(num_runs):
        temp = 0.3 + run * 0.15  # 0.3, 0.45, 0.6, 0.75, 0.9
        prompt = build_sequence_prompt(rows, num_context=12)
        response = query_llm_plain(prompt, max_tokens=40, temp=temp)
        nums = parse_numbers(response)
        
        print(f"    Run {run+1} (temp={temp:.1f}): {nums[:10]}... <- {response[:50]}")
        
        for n in nums:
            if 1 <= n <= 49:
                all_votes[n] = all_votes.get(n, 0) + 1
    
    # 取出现最多的6个为平码，再加出现最多的做特码
    sorted_nums = sorted(all_votes.items(), key=lambda x: -x[1])
    
    result_nums = []
    result_spc = None
    
    for n, v in sorted_nums:
        if len(result_nums) < 6:
            result_nums.append(n)
        else:
            result_spc = n
            break
    
    # 如果不够，补齐
    while len(result_nums) < 6:
        for n in range(1, 50):
            if n not in result_nums:
                result_nums.append(n)
                break
    
    if not result_spc:
        for n in range(1, 50):
            if n not in result_nums:
                result_spc = n
                break
    
    return result_nums, result_spc

if __name__ == "__main__":
    print("🦦 LLM 彩票预测引擎 (纯文本模式)")
    print("=" * 50)
    
    rows = load_data()
    print(f"📊 数据: {len(rows)}期")
    print(f"  上期: {rows[-1][0]} + T{rows[-1][1]}")
    
    print("\n🚀 启动 llama-server...")
    if start_server():
        print("  ✅ OK")
    else:
        print("  ❌ 失败!")
        sys.exit(1)
    
    try:
        print("\n🔮 单次预测 (temp=0.5):")
        nums, spc, raw = predict_single(temp=0.5)
        if nums:
            print(f"  🎯 {nums} + T{spc}")
        else:
            print(f"  ⚠️ 原始: {raw}")
        
        print("\n🔮 多次综合预测 (5次):")
        ens_nums, ens_spc = predict_ensemble(num_runs=5)
        print(f"  🎯 {ens_nums} + T{ens_spc}")
        
    finally:
        stop_server()
