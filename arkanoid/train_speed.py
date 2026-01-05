import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. 讀取資料
path = os.path.dirname(__file__)
data_path = os.path.join(path, "data.pickle")
target_path = os.path.join(path, "target.pickle")

if not os.path.exists(data_path):
    print(" 找不到 data.pickle，請先執行遊戲進行蒐集！")
    exit()

with open(data_path, "rb") as f:
    data = pickle.load(f)
with open(target_path, "rb") as f:
    target = pickle.load(f)

print(f"📂 載入數據... 共 {len(data)} 筆")

# 2. 轉換格式
X = np.array(data)
y = np.array(target)

# 3. 設定隨機森林參數
# n_estimators=100: 召喚 100 棵決策樹來投票
# max_depth=20: 限制樹的深度，避免它過度死記硬背 (Overfitting)
print(" 開始訓練隨機森林模型...")
model = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1)
model.fit(X, y)

# 4. 儲存模型
model_path = os.path.join(path, "rf_model.pickle")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"🎉 訓練完成！模型已儲存至: {model_path}")
print("現在可以執行 ml_play.py 進行測試了。")