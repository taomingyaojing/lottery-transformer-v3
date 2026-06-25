"""
彩票预测网关 - RESTful API
提供模型预测和查询服务

启动:
  python3 gateway.py

API:
  GET  /health        - 健康检查
  POST /predict       - 运行预测（V3 / 增强 / 集成）
  GET  /history       - 历史预测记录
  GET  /backtest      - 回测统计
"""

import json, os, sys, csv
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime

# 配置
PORT = int(os.environ.get("PORT", 8080))
MODEL_DIR = os.environ.get("MODEL_DIR", "./models/v3")
ENSEMBLE_DIR = os.environ.get("ENSEMBLE_DIR", "./ensemble_output")
DATA_PATH = os.environ.get("DATA_PATH", "./data/lottery_history.csv")
PRED_DIR = os.environ.get("PRED_DIR", "./predictions")

class LotteryAPI(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)
        
        if path == "/health":
            self.send_json({
                "status": "ok",
                "model": "V3",
                "port": PORT,
                "time": datetime.now().isoformat(),
            })
        elif path == "/history":
            self.handle_history(params)
        elif path == "/backtest":
            self.handle_backtest(params)
        else:
            self.send_json({"error": "not found"}, 404)
    
    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len else b"{}"
        
        try:
            params = json.loads(body)
        except:
            params = {}
        
        if path == "/predict":
            self.handle_predict(params)
        else:
            self.send_json({"error": "not found"}, 404)
    
    def handle_predict(self, params):
        mode = params.get("mode", "v3")
        
        try:
            if mode == "v3":
                result = self.run_v3_prediction()
            elif mode == "enhanced":
                result = self.run_enhanced_prediction()
            elif mode == "ensemble":
                result = self.run_ensemble_prediction()
            else:
                self.send_json({"error": f"unknown mode: {mode}"}, 400)
                return
            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)
    
    def handle_history(self, params):
        limit = int(params.get("limit", [10])[0])
        results = []
        if os.path.isdir(PRED_DIR):
            files = sorted(os.listdir(PRED_DIR), reverse=True)[:limit]
            for f in files:
                if f.endswith(".json"):
                    with open(os.path.join(PRED_DIR, f)) as fp:
                        results.append(json.load(fp))
        self.send_json({"predictions": results, "count": len(results)})
    
    def handle_backtest(self, params):
        n = int(params.get("n", [100])[0])
        # 读取最近N期进行简单回测
        rows = []
        with open(DATA_PATH) as f:
            reader = csv.reader(f)
            next(reader)
            for r in reader:
                vals = r[2::4]
                nums = [int(v) for v in vals[:7] if v.strip()]
                if len(nums) == 7:
                    rows.append(nums)
        
        total = len(rows)
        start = max(0, total - n - 8)
        test = rows[start:total]
        
        self.send_json({
            "total_draws": total,
            "test_range": f"{len(test)} draws",
        })
    
    def run_v3_prediction(self):
        """运行V3预测（调用infer.py）"""
        import subprocess
        result = subprocess.run(
            [sys.executable, "infer.py", "--json"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        return {"error": result.stderr}
    
    def run_ensemble_prediction(self):
        """读取已有集成预测结果"""
        path = os.path.join(ENSEMBLE_DIR, "latest_ensemble_prediction.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        
        # 如果不存在，尝试读取各模型历史
        predictions = {}
        for name in ["model_a_deep", "model_b_wide", "model_c_lstm"]:
            hp = os.path.join(ENSEMBLE_DIR, f"{name}_history.json")
            if os.path.exists(hp):
                with open(hp) as f:
                    predictions[name] = json.load(f)
        
        return {
            "status": "partial",
            "ensemble_available": False,
            "models": {k: {"best_loss": v.get("best_val_loss")} for k, v in predictions.items()},
        }
    
    def run_enhanced_prediction(self):
        return self.run_v3_prediction()
    
    def send_json(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode())

if __name__ == "__main__":
    print(f"🚀 Lottery Prediction Gateway")
    print(f"   Listening on http://0.0.0.0:{PORT}")
    print(f"   Endpoints:")
    print(f"     GET  /health")
    print(f"     POST /predict")
    print(f"       modes: v3, enhanced, ensemble")
    print(f"     GET  /history")
    print(f"     GET  /backtest")
    print()
    
    server = HTTPServer(("0.0.0.0", PORT), LotteryAPI)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 服务已关闭")
        server.server_close()
