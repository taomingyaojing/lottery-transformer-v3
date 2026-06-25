#!/usr/bin/env python3
"""
🦦 每日彩票数据更新 + 一键预测 (03:00执行)
流程: web_fetch抓取 → scp到远程 → 更新csv → 跑Ensemble预测
"""
import subprocess, sys, os, json, urllib.request, re

DATA_URL = "https://hqkeajgv.qddrj-5jnqx-zktazn.work:16677/kj/3/2026.html"
REMOTE = "ubuntu@43.160.255.252"
REMOTE_SCRIPT = "~/workspace/scripts/data/update.py"
PREDICT_SCRIPT = "~/workspace/run_all_prediction_v2.py"

def fetch_page_text(url):
    """用类似readability方式获取纯文本"""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except:
        # 如果直接请求是乱码，可能需要调用web_fetch
        print("⚠️ 直接请求可能被加密，尝试备用方式...")
        return None
    
    # 尝试从HTML中提取纯文本数字
    # 如果html包含乱码，返回None让调用者处理
    if any(ord(c) > 127 for c in html[:100]) and "期" not in html[:1000]:
        return None  # 乱码
    return html

def main():
    print("=" * 60)
    print("🦦 每日彩票数据更新 + 预测")
    print("=" * 60)
    
    # Step 1: 获取数据
    print("\n📡 步骤1: 获取最新开奖数据...")
    
    # 尝试直接请求
    text = fetch_page_text(DATA_URL)
    
    if text and "第" in text:
        # 保存文本
        with open("/tmp/lottery_fresh_data.txt", "w") as f:
            f.write(text)
        print("   ✅ 数据获取成功")
    else:
        print("   ❌ 数据获取失败，跳过更新")
        # 仍然尝试用已有数据跑预测
        text = None
    
    # Step 2: 更新远程csv
    print("\n📁 步骤2: 更新远程数据文件...")
    if text:
        # 传到远程
        subprocess.run([
            "scp", "/tmp/lottery_fresh_data.txt",
            f"{REMOTE}:~/workspace/lottery_fresh_data.txt"
        ], capture_output=True)
        
        # 在远程执行更新
        result = subprocess.run([
            "ssh", REMOTE,
            f"cd ~/workspace && python3 {REMOTE_SCRIPT} lottery_fresh_data.txt 2>&1"
        ], capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print(f"   ⚠️ {result.stderr}")
    else:
        print("   ⏭️ 跳过更新")
    
    # Step 3: 跑预测
    print("\n🔮 步骤3: 运行三模型Ensemble预测...")
    result = subprocess.run([
        "ssh", REMOTE,
        f"cd ~/workspace && python3 {PREDICT_SCRIPT} 2>&1"
    ], capture_output=True, text=True, timeout=600)
    print(result.stdout[-1500:] if len(result.stdout) > 1500 else result.stdout)
    if result.stderr:
        print(f"   ⚠️ {result.stderr}")
    
    print("\n✅ 完成! 🦦")

if __name__ == "__main__":
    main()
