import pickle
import os
import json
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, "mlgame_data.txt")
model_path = os.path.join(dir_path, "my_rf_model.pickle")

# 1. 讀取數據
print(f"正在讀取數據: {data_path}")
data_moves = []
data_nones = []

if not os.path.exists(data_path):
    print(f"錯誤: 找不到 {data_path}")
    exit()

with open(data_path, "r") as f:
    for line in f:
        try:
            row = json.loads(line)
            # 分類：是「移動」還是「不動」
            if row["action"] == "NONE":
                data_nones.append(row)
            else:
                data_moves.append(row)
        except:
            continue

print(f"原始資料: 移動(Moves)={len(data_moves)}, 不動(NONE)={len(data_nones)}")

# 2. 強制平衡數據 (關鍵步驟！)
# 我們只保留跟「移動」一樣多的「不動」，其他的 NONE 全部丟掉
# 這樣 AI 就不會覺得「不動」是常態了
if len(data_nones) > len(data_moves):
    print("⚠️ 偵測到 NONE 太多，正在自動刪減，讓 AI 更積極...")
    data_nones = random.sample(data_nones, len(data_moves))

final_data = data_moves + data_nones
random.shuffle(final_data) # 打亂順序

print(f"最終訓練資料: {len(final_data)} 筆 (已平衡)")

# 3. 整理特徵
x_data = []
y_data = []

for row in final_data:
    feature = [
        row["ball_x"],
        row["ball_y"],
        row["ball_speed_x"],
        row["ball_speed_y"],
        row["paddle_x"]
    ]
    x_data.append(feature)
    y_data.append(row["action"])

# 4. 訓練模型
model = RandomForestClassifier(n_estimators=100, n_jobs=-1)

print("開始訓練...")
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)
model.fit(x_train, y_train)

# 5. 驗收
score = model.score(x_test, y_test)
print(f"訓練完成！準確率: {score * 100:.2f}%")
print("這個準確率如果只有 80%~90% 是正常的，代表它現在學會『糾結』了，這比學會『偷懶』好！")

# 6. 存檔
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"模型已儲存: {model_path}")