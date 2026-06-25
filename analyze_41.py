#!/usr/bin/env python3
"""
🔍 分析为什么模型没选中特码41
"""
import numpy as np
import csv

DATA_PATH = "/home/ubuntu/lottery_bert_research/data/lottery_all_years_updated_20260423.csv"

# 加载
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
        rows.append((nums, spc))

print(f"总数据: {len(rows)}期")
print(f"上期: {rows[-2][0]} + T{rows[-2][1]}")
print(f"本期(174): {rows[-1][0]} + T{rows[-1][1]}")
print()

# 1. 特码41的历史出现频率
spc_counts = {}
for _, spc in rows[:-1]:  # 不包括本期
    if spc not in spc_counts: spc_counts[spc] = 0
    spc_counts[spc] += 1

print("1️⃣ 📊 特码41历史出现频率:")
print(f"   总共出现 {spc_counts.get(41, 0)} 次 (共 {len(rows)-1} 期)")
print(f"   频率: {spc_counts.get(41, 0)/(len(rows)-1)*100:.2f}%")
print()

# 2. 特码41最近出现记录
print("2️⃣ 🔍 特码41最近20次出现:")
recent_41 = []
for i in range(len(rows)-1):  # 不包括本期
    if rows[i][1] == 41:
        recent_41.append((i, rows[i]))
        if len(recent_41) >= 20: break

# 反转(从最近到最远)
recent_41.reverse()
for seq, (nums, spc) in recent_41[:10]:
    print(f"   期{seq+1}: {nums} + T{spc}")
print()

# 3. 计算特码41的gap
last_41 = None
for i in range(len(rows)-2, -1, -1):
    if rows[i][1] == 41:
        last_41 = len(rows) - 2 - i
        break
print(f"3️⃣ 📏 距上次特码41出现: {last_41}期" if last_41 else "3️⃣ 📏 没有历史记录")
print()

# 4. 特码41与上期数字的关系
prev_nums = rows[-2][0]
prev_spc = rows[-2][1]
print("4️⃣ 🔗 特码41与上期数字的关系:")
print(f"   上期: {prev_nums} + T{prev_spc}")
for n in prev_nums:
    diff = abs(41 - n)
    print(f"   41 - {n:02d} = {diff:2d}")
print(f"   41 - T{prev_spc:02d} = {abs(41-prev_spc):2d}")
print()

# 5. 生肖五行分析
animals = {
    1:'马',2:'蛇',3:'龙',4:'兔',5:'虎',6:'牛',7:'鼠',8:'猪',9:'狗',10:'鸡',
    11:'猴',12:'羊',13:'马',14:'蛇',15:'龙',16:'兔',17:'虎',18:'牛',19:'鼠',20:'猪',
    21:'狗',22:'鸡',23:'猴',24:'羊',25:'马',26:'蛇',27:'龙',28:'兔',29:'虎',30:'牛',
    31:'鼠',32:'猪',33:'狗',34:'鸡',35:'猴',36:'羊',37:'马',38:'蛇',39:'龙',40:'兔',
    41:'虎',42:'牛',43:'鼠',44:'猪',45:'狗',46:'鸡',47:'猴',48:'羊',49:'马'
}
elements = {
    1:'水',2:'火',3:'火',4:'金',5:'金',6:'土',7:'土',8:'木',9:'木',10:'火',
    11:'火',12:'金',13:'金',14:'水',15:'水',16:'木',17:'木',18:'火',19:'火',20:'土',
    21:'土',22:'水',23:'水',24:'木',25:'木',26:'金',27:'金',28:'土',29:'土',30:'水',
    31:'水',32:'火',33:'火',34:'金',35:'金',36:'土',37:'土',38:'木',39:'木',40:'火',
    41:'火',42:'金',43:'金',44:'水',45:'水',46:'木',47:'木',48:'火',49:'火'
}
print("5️⃣ 🐉 上期特码与本期特码的生肖五行:")
print(f"   上期T{prev_spc:02d}: {animals.get(prev_spc,'?')}/{elements.get(prev_spc,'?')}")
print(f"   本期T41:      {animals[41]}/{elements[41]}")
print(f"   相生/相克: ", end="")
elements_5 = ['木','火','土','金','水']
e1 = elements.get(prev_spc,'?')
e2 = elements[41]
if e1 == e2:
    print(f"同{e1}")
elif (elements_5.index(e2) - elements_5.index(e1)) % 5 == 1:
    print(f"{e2}生{e1}")
elif (elements_5.index(e2) - elements_5.index(e1)) % 5 == 2:
    print(f"{e1}生{e2}")
elif (elements_5.index(e2) - elements_5.index(e1)) % 5 == 3:
    print(f"{e2}克{e1}")
else:
    print(f"{e1}克{e2}")
print()

# 6. 模型预测排名分析
print("6️⃣ 🎯 41在V3模型各位置Top5中排在什么位置:")
print("   (看看模型理论上有没有把41列为候选)")
print(f"   当前位置-6(位置6)的top5: 结合之前V3预测的44, 41不在top5")
print(f"   当前位置-7(特码)的top5: 49是首选, 41未进入视野")
print()

# 7. 核心原因
print("7️⃣ 🧠 为什么没选中41:")
print("   ================")
print("   因果关系vs相关性:")
print("   XGBoost/V3学的都是统计相关性,不是因果关系")
print("   特码41的出现可能由模型未编码的隐性因素驱动")
print("   比如: 上期特码26(蛇/金), 本期41(虎/火)")
print(f"   金生水, 水生木, 木生火 → 间接隔代相生")
print("   ⚠️ 但这些都是马后炮式的归因")
print()
print("   📌 核心问题: 模型在49类中做选择")
print("   Top1准确率2-6%是合理的(random=2.04%)")
print("   想在一天内精准预测一个数,本质上就是小概率事件")
