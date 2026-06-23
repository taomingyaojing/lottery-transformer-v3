# 彩票Transformer V3 — 獭獭预测引擎 🦦

## 模型家族

| 模型 | 文件 | 说明 |
|------|------|------|
| V3 Transformer | train.py / infer.py | 6层Transformer, 512维, 序列模式 |
| XGBoost | ft_xgb_lottery.py | 707维统计特征, 7个多分类器 |
| Ensemble | run_all_prediction.py | 一键全模型 + 加权融合 |

## 快速开始

```bash
# 一键预测（需先训练好模型）
python3 run_all_prediction.py

# V3训练
python3 train.py

# V3推理
python3 infer.py

# XGBoost训练
python3 ft_xgb_lottery.py
```

## 依赖

torch / xgboost / scikit-learn / numpy / pandas

## 输出格式

```json
{
  "engine": "Ensemble_v1",
  "ensemble": {
    "nums": [35, 29, 3, 10, 17, 32],
    "spc": 49
  },
  "individual_models": {
    "v3": { "nums": [...], "spc": 49 },
    "xgb": { "nums": [...], "spc": 49 }
  }
}
```
