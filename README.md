# 🎯 Lottery Transformer V3

基于Transformer的彩票预测模型（6平码+1特码），支持多窗口投票集成和HTTP API网关。

## 架构

```
V3 核心模型: Transformer 8层 (512d, 8头, FFN 2048)
  输入: 343维 one-hot 编码 (7个位置 × 49个号码)
  输出: 7个位置 × 49类别 softmax

增强版本: 冷热度后处理融合 (α=0.15/0.20)

集成版本: 4模型投票
  A-Deep:   Transformer 8层 (17M)
  B-Wide:   Transformer 6层 + 宽FFN (30M)  
  C-LSTM:   BiLSTM 4层 (24M)
  D-CNN:    CNN + Attention (2M)
```

## 文件结构

```
lottery-v3-open/
├── train.py             # V3核心训练（数据增强）
├── train_enhanced.py    # V3+增强训练（冷热度融合）
├── infer.py             # V3+推理 + 系统回测
├── ensemble_train.py    # 4模型集成训练
├── gateway.py           # RESTful API 预测网关
├── data/
│   └── lottery_history.csv  # 历史开奖数据 (31列)
├── model_output/        # 训练好的模型权重
├── ensemble_output/     # 集成模型权重
├── predictions/         # 每日预测结果
└── requirements.txt     # 依赖
```

## 数据格式

CSV 31列，每4列一组（号码,颜色,生肖,五行）：
```
period,date,number_1,color_1,zodiac_1,element_1,...,special_number,...,total_numbers
```

## 使用方法

```bash
# 训练
python3 train.py

# 推理+预测
python3 infer.py

# 集成训练
python3 ensemble_train.py

# 启动预测网关
python3 gateway.py
#   GET  /health        - 健康检查
#   POST /predict       - 运行预测 (v3/enhanced/ensemble)
#   GET  /history       - 历史预测
#   GET  /backtest      - 回测统计
```

## 依赖

- Python 3.10+
- PyTorch 2.0+
- NumPy

## 声明

本项目仅供学习研究使用。
