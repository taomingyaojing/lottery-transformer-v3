#!/usr/bin/env python3
"""
📡 彩票数据自动更新脚本
通过OpenClaw web_fetch能力获取数据，更新到远程服务器
"""
import csv, os, sys, re, json, urllib.request

# 配置
DATA_URL = "https://hqkeajgv.qddrj-5jnqx-zktazn.work:16677/kj/3/2026.html"
CSV_PATH = "/home/ubuntu/lottery_bert_research/data/lottery_all_years_updated_20260423.csv"

# 从本地文件读取上一次web_fetch的结果（由定时任务触发时提供）
# 或者直接从标准输入获取html文本

def parse_periods_from_text(text):
    """从纯文本格式解析开奖数据"""
    periods = []
    
    # 格式: YYYY年MM月DD日 第NNN期 + 7个数字
    # 用正则匹配日期行
    lines = text.strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # 匹配 "2026年06月22日 第173期" 这类行
        m = re.search(r'(\d{4})年(\d{2})月(\d{2})日\s*第(\d+)期', line)
        if m:
            year, month, day, period = m.groups()
            
            # 从下一行开始找数字 (每行一个数字)
            nums = []
            j = i + 1
            while j < len(lines) and len(nums) < 7:
                l = lines[j].strip()
                # 匹配纯数字行：行首是1-2位数字，后面可能跟生肖五行
                nm = re.match(r'^(\d{1,2})', l)
                if nm:
                    n = int(nm.group(1))
                    if 1 <= n <= 49:
                        nums.append(n)
                j += 1
            
            if len(nums) == 7:
                periods.append({
                    "date": f"{year}-{month}-{day}",
                    "period": int(period),
                    "nums": nums[:6],
                    "spc": nums[6]
                })
            i = j
        else:
            i += 1
    
    return periods

def load_existing_csv(path):
    """加载已有数据"""
    rows = []
    try:
        with open(path) as f:
            reader = csv.reader(f)
            h = next(reader)
            for r in reader:
                rows.append(r)
    except:
        pass
    return rows

def update_csv(periods, existing_rows, path):
    """合并新数据到csv"""
    existing_dates = set()
    max_id = 0
    for row in existing_rows:
        if len(row) > 1:
            existing_dates.add(row[1])
        try:
            max_id = max(max_id, int(row[0]))
        except:
            pass
    
    new_rows = []
    for p in sorted(periods, key=lambda x: x["date"]):
        if p["date"] in existing_dates:
            continue
        
        max_id += 1
        row = [str(max_id), p["date"]]
        for idx, n in enumerate(p["nums"]):
            col_pos = 2 + idx * 4
            while len(row) < col_pos:
                row.append("")
            row.append(str(n))
        while len(row) < 26:
            row.append("")
        row.append(str(p["spc"]))
        new_rows.append(row)
    
    if new_rows:
        # 追加写入
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            for row in new_rows:
                writer.writerow(row)
    
    return new_rows

def main():
    print("🦦 彩票数据更新")
    
    # 如果有文件参数，从文件读
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            text = f.read()
    else:
        # 从标准输入读
        text = sys.stdin.read()
    
    if not text.strip():
        print("❌ 没有输入数据!")
        sys.exit(1)
    
    periods = parse_periods_from_text(text)
    print(f"📊 解析到 {len(periods)} 期")
    
    if not periods:
        print("❌ 没有解析到任何期次!")
        sys.exit(1)
    
    latest = periods[-1]
    print(f"📅 最新: {latest['date']} 第{latest['period']}期")
    print(f"   {latest['nums']} + T{latest['spc']}")
    
    existing = load_existing_csv(CSV_PATH)
    print(f"📁 已有 {len(existing)} 条记录")
    
    new_rows = update_csv(periods, existing, CSV_PATH)
    
    if new_rows:
        print(f"✅ 新增 {len(new_rows)} 期数据")
        for r in new_rows:
            print(f"   {r[0]}. {r[1]}: {r[2] or '-'} {r[6] or '-'} {r[10] or '-'} {r[14] or '-'} {r[18] or '-'} {r[22] or '-'} + T{r[26] if len(r)>26 else '-'}")
    else:
        print("✅ 数据已是最新")
    
    # 输出JSON供后续使用
    result = {
        "latest": {
            "date": latest["date"],
            "period": latest["period"],
            "nums": latest["nums"],
            "spc": latest["spc"]
        },
        "added": len(new_rows),
        "total": len(existing) + len(new_rows)
    }
    print(f"\n{json.dumps(result)}")

if __name__ == "__main__":
    main()
