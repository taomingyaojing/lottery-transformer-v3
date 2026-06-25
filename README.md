# 🦦 彩票Transformer V3 — 獭獭预测引擎

三模型集成（Ensemble）彩票预测系统，基于历史开奖数据，通过 V3 Transformer + XGBoost + LLM 加权融合给出预测。

## 🏗 模型家族

| 模型 | 文件 | 说明 |
|------|------|------|
| **V3 Transformer** | `train.py` / `infer.py` | 6层Transformer, 512维, 序列模式 + 冷热度后处理补偿 |
| **XGBoost** | `ft_xgb_lottery.py` | 707维统计特征, 7个多分类器 |
| **LLM (Qwen2.5-1.5B)** | `llm_lottery_infer_v2.py` | 纯文本序列模式推理, 多次采样投票 |
| **V2 Ensemble** | `run_all_prediction_v2.py` | 三模型加权融合预测引擎 |
| **集成训练** | `ensemble_train.py` | 多模型并行训练 + 投票集成 |

## 🔄 自动化管线

| 脚本 | 功能 |
|------|------|
| `daily_lottery_pipeline.py` | 每日开奖数据抓取 -> CSV更新 -> 三模型预测 -> 结果归档 |
| `update_lottery_data.py` | 从开奖网页解析数据并追加到历史CSV |
| `infer.py` | V3增强推理（softmax + 冷热度补偿） |

## 📊 数据编码

每期开奖编码为 **343维 one-hot 向量**：
- 6个平码 * 49 = 294维（每个数字一个位置）
- 1个特码 * 49 = 49维
- 输入序列长度 = 8期（多窗口voting：3/5/8/10期滑动）

## 🧠 推理策略

### V3 Transformer
```
多窗口滑动 -> 6层Transformer编码 -> 7个分类头(6平+1特) -> softmax
-> 统计近10/30/60期频率 + gap(距上次出现期数)
-> 冷热度评分融合修正输出
```

### XGBoost
```
707维统计特征（奇偶比、和值、跨度、生肖、五行、邻号等）
-> 7个多分类器各自预测
```

### LLM (Qwen2.5-1.5B 4-bit量化)
```
最近12期序列写成文本 -> llama.cpp推理 -> 续写"Next line:"
-> 5次采样(温度0.3~0.9) -> 投票选出出现最多的数字
```

### Ensemble 融合
```
V3结果 * 权重 + XGBoost结果 * 权重 + LLM结果 * 权重
-> 加权融合 -> 不重复选取Top6平码 + Top1特码
```

## 🔬 冷热度补偿（后处理）

- **热号**（gap <= 8期）：高频率，给予高分
- **温号**（中间地带）：适中boost
- **冷号**（gap >= 20期）：低频但有爆发潜力，给予补偿性加分
- 特码位置补偿权重略高（alpha=0.20 vs 平码alpha=0.15）

## 🔧 配置

### 硬性依赖
```bash
torch >= 1.13
xgboost >= 1.7
scikit-learn >= 1.2
numpy >= 1.22
pandas >= 1.4
```

### LLM 额外依赖
- [llama.cpp](https://github.com/ggerganov/llama.cpp) 构建的 `llama-server`
- Qwen2.5-1.5B GGUF 4-bit量化模型

## 🚀 快速开始

```bash
# 1. 训练V3模型
python3 models/v3/train.py

# 2. 训练XGBoost
python3 models/xgb/train.py

# 3. 一键预测（需先训练好模型）
python3 ensemble/predict_v1.py

# 4. V2 Ensemble（含LLM）
python3 ensemble/predict.py

# 5. 自动化流程
python3 pipeline/daily.py

# 6. 数据更新
python3 scripts/data/update.py <数据文件>
```

## 📤 输出格式

```json
{
  "engine": "Ensemble_v2",
  "date": "2026-06-25",
  "ensemble": {
    "nums": [7, 19, 29, 30, 25, 28],
    "spc": 26
  },
  "individual_models": {
    "v3": { "nums": [...], "spc": 49 },
    "xgb": { "nums": [...], "spc": 49 },
    "llm": { "nums": [...], "spc": 49, "run_count": 5 }
  },
  "ensemble_weights": { "v3": 0.4, "xgb": 0.3, "llm": 0.3 }
}
```

## 📁 项目结构

```
.
├── train.py                     # V3 Transformer 训练
├── infer.py                     # V3 增强推理（含冷热度补偿）
├── ft_xgb_lottery.py            # XGBoost 训练
├── run_all_prediction.py        # V1 Ensemble (V3 + XGBoost)
├── run_all_prediction_v2.py     # V2 Ensemble (V3 + XGBoost + LLM)
├── ensemble_train.py            # 多模型并行集成训练
├── daily_lottery_pipeline.py    # 每日自动化管线
├── update_lottery_data.py       # 开奖数据更新
├── llm_lottery_infer_v2.py      # LLM 序列预测（纯文本模式）
├── llm_lottery_infer.py         # LLM 推理（旧版）
├── llm_lottery_proposal.py      # LLM 方案提案
├── qwen_lora_train.py           # Qwen LoRA 微调
├── qwen_lora_train_v2.py        # Qwen LoRA 微调 v2
├── prepare_qwen_data.py         # Qwen 训练数据准备
├── analyze_41.py                # 41号分析
├── noise_v3.py                  # 噪声测试 v3
├── noise_test.py                # 噪声测试
├── shift_test.py                # 偏移测试
├── gateway.py                   # 预测网关 API
├── requirements.txt
└── README.md
```

## 📜 历史提交

- `a3faf00` ➕ V2 Ensemble: V3 + XGBoost + LLM 三模型集成
- `40d1c90` ➕ 添加 XGBoost 预测引擎 + Ensemble 一键预测脚本
- `79ff322` 📝 更新README，添加网关使用说明
- `d089c70` ➕ 添加预测网关 API
- `0ada815`  Lottery Transformer V3 - 初始提交
