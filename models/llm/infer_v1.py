#!/usr/bin/env python3
"""
🦦 LLM 彩票预测 — Qwen2.5-1.5B via llama.cpp
"""
import json, csv, subprocess, os, signal, sys, time, urllib.request, urllib.error, re

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
    cmd = [
        LLAMA_SERVER_PATH,
        "-m", MODEL_PATH,
        "--port", str(PORT),
        "--host", "127.0.0.1",
        "-ngl", "0",
        "-c", "8192",
        "--no-mmap",
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

def query_llm(prompt, max_tokens=64, temp=0.3):
    data = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temp,
        "stop": ["<|im_end|>", "\n\n"],
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result.get("content", "").strip()
    except Exception as e:
        return f"ERROR: {e}"

def build_prompt(rows, num_context=12):
    """用更多上下文 + 更明确的格式要求"""
    recent = rows[-num_context:] if len(rows) >= num_context else rows
    
    lines = []
    for i, (nums, spc) in enumerate(recent):
        nums_str = "-".join(f"{n:02d}" for n in nums)
        spc_str = f"{spc:02d}" if spc else "??"
        lines.append(f"第{i+1}期: {nums_str} + {spc_str}")
    
    prompt = f"""<|im_start|>system
你是一个彩票号码预测系统。号码范围: 01-49。分析历史数据规律后，预测下一期的号码。
输出格式:
平码: 01-02-03-04-05-06
特码: 07
只输出两行，不要解释。
<|im_end|>
<|im_start|>user
最近{len(recent)}期数据:
{"\n".join(lines)}

预测下一期的7个号码:
<|im_end|>
<|im_start|>assistant
"""
    return prompt

def parse_response(text):
    """解析LLM输出"""
    # 提取所有1-49的数字
    nums = [int(x) for x in re.findall(r'\b([1-9]|[1-4][0-9])\b', text)]
    # 过滤掉>49的
    nums = [n for n in nums if 1 <= n <= 49]
    
    if len(nums) < 7:
        return None, None
    
    # 前6个作为平码，最后一个作为特码
    result_nums = nums[:6]
    result_spc = nums[-1]
    
    return result_nums, result_spc

def predict(rows):
    prompt = build_prompt(rows, num_context=12)
    print("  🤖 调用 LLM 推理...")
    start = time.time()
    response = query_llm(prompt, max_tokens=64, temp=0.3)
    elapsed = time.time() - start
    print(f"  ⏱️  {elapsed:.1f}s")
    print(f"  📝 原始输出: {response}")
    nums, spc = parse_response(response)
    return nums, spc, response

if __name__ == "__main__":
    print("🦦 LLM 彩票预测引擎")
    print("=" * 50)
    print("📊 加载数据...")
    rows = load_data()
    print(f"  总期数: {len(rows)}")
    print(f"  上期: {rows[-1][0]} + T{rows[-1][1]}")
    print("\n🚀 启动 llama-server...")
    if start_server():
        print("  ✅ 服务器启动成功!")
    else:
        print("  ❌ 服务器启动失败!")
        sys.exit(1)
    try:
        print("\n🔮 预测中...")
        nums, spc, raw = predict(rows)
        if nums and spc:
            print(f"\n🎯 LLM预测:")
            print(f"  平码: {nums}")
            print(f"  特码: {spc}")
        else:
            print(f"\n⚠️  解析失败, 原始输出: {raw}")
    finally:
        stop_server()
    print("\n⏱️ 完成! 🦦")
