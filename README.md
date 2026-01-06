# 機器學習與實作
## 需求-功能
<img width="1544" height="793" alt="image" src="https://github.com/user-attachments/assets/d33c934a-0439-4b80-8f05-c63211e08174" />
<img width="1155" height="586" alt="image" src="https://github.com/user-attachments/assets/f86b6605-7411-4c53-9d28-a0aec4b05347" />

## 需求-pingpong介面
<img width="795" height="796" alt="image" src="https://github.com/user-attachments/assets/0aacc3d6-a6d3-498e-87ad-2d86b56a6ef4" />

## 需求-打磚塊介面
<img width="692" height="499" alt="image" src="https://github.com/user-attachments/assets/40b9657c-09ae-43ec-b9eb-8cd628fade0c" />

## 需求-限制
遊戲區域：200×500像素，左上角為原點
遊戲物件：
球：  5*5像素藍色方形，每影格移動為(±7，±7)，坐標範圍(0,0)~(195,403)。
            球從板子所在位置發出，可選擇往左往右發球，若在150影格內沒發球，則會自動往隨機兩方向發球。

板子：40×5像素綠色長方形，每影格移動為(±5，0)，坐標範圍(0,400)~(160,400)，初始位置在(75,400)。

切球機制：
球的X軸方向速度會因接球時板子之移動方向而改變：
          1.若板子與球移動方向相同，則球的X軸方向速度會增加±10，可以一次打掉
          2.若板子不動，則球的X軸方向速度會回復為±7
語言:Python(AI訓練)

環境版本:
Python:3.9

模組版本:

作業系統： 
Windows 10 專業版64位元

# 乒乓球
## 分析
<img width="2000" height="503" alt="image" src="https://github.com/user-attachments/assets/99a2d3eb-ecb2-400f-8fc0-6cd53537500c" />

## 架構
<img width="1292" height="827" alt="image" src="https://github.com/user-attachments/assets/be58d2a9-ecbd-499b-aeab-f86ea7971f8f" />
<img width="300" height="886" alt="image" src="https://github.com/user-attachments/assets/f032a1d7-eefd-43ac-82d7-020112aa92bc" />

## API
### 專案 : 資料收集與規則階段 (ml_play_rule.py)
<img width="1076" height="783" alt="image" src="https://github.com/user-attachments/assets/f7d36e7e-dcf3-467d-b9d0-8bcda441fb53" />

### 專案 : 模型訓練階段(train.py)
<img width="1076" height="446" alt="image" src="https://github.com/user-attachments/assets/b5814401-05e5-45ce-841c-9fafcb018c7a" />

### 專案 : AI推論與遊玩階段 (ml_play.py)
<img width="1076" height="703" alt="image" src="https://github.com/user-attachments/assets/1398db9b-84ed-4dd6-b3aa-8c4e8ed7eb2e" />

## API&程式碼
<img width="1076" height="442" alt="image" src="https://github.com/user-attachments/assets/cdea82e6-ebe6-4347-9d91-f5ec99c3934c" />

    def update(self, scene_info, *args, **kwargs):
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"

        ball_x = scene_info["ball"][0]
        ball_y = scene_info["ball"][1]
        speed_x = scene_info["ball_speed"][0]
        speed_y = scene_info["ball_speed"][1]
        
        if self.side == "1P":
            paddle_x = scene_info["platform_1P"][0]
            target_y = 415
        else:
            paddle_x = scene_info["platform_2P"][0]
            target_y = 85

        # --- 判斷球是否朝我飛來 (關鍵變數) ---
        is_coming = (self.side == "1P" and speed_y > 0) or (self.side == "2P" and speed_y < 0)

        # --- 預判落點邏輯 ---
        if is_coming:
            steps = (target_y - ball_y) / speed_y
            pred_x = ball_x + (speed_x * steps)
            while pred_x < 0 or pred_x > 200:
                if pred_x > 200:
                    pred_x = 200 - (pred_x - 200)
                else:
                    pred_x = -pred_x
        else:
            pred_x = 100
            
         #--- 移動邏輯 ---
        command = "NONE"
        paddle_center = paddle_x + 20
        
        # 【修改這裡】把 10 改成 2，讓老師變得神經質，一點點偏差都要修正
        if paddle_center < pred_x - 2: 
            command = "MOVE_RIGHT"
        elif paddle_center > pred_x + 2:
            command = "MOVE_LEFT"

        # --- 【關鍵修改】只存有意義的數據 ---
        # 條件 1: 球必須在動 (排除發球前)
        # 條件 2: 球必須是朝我飛來的 (is_coming 為 True)
       # 只存 1P 的資料，且球是朝我飛來的
       # 只要球在動，我都存！這樣 AI 才知道球打出去後要「回中間準備」
        if self.side == "1P" and (speed_x != 0 or speed_y != 0):
            current_step_data = {
                "ball_x": ball_x,
                "ball_y": ball_y,
                "ball_speed_x": speed_x,
                "ball_speed_y": speed_y,
                "paddle_x": paddle_x,
                "action": command
            }
            self.data.append(current_step_data)

        # 自動存檔機制 (避免視窗關閉沒存到)
        if len(self.data) >= 1000:
            self.flush_data()

        return command

<img width="1076" height="292" alt="image" src="https://github.com/user-attachments/assets/09f99671-49fa-438d-b980-22d47e987deb" />

    def reset(self):
        self.flush_data()
        print(f"[{self.side}] 遊戲結束，精華數據已儲存！")

    def flush_data(self):
        if not self.data:
            return
        file_path = os.path.join(self.dir_path, "mlgame_data.txt")
        with open(file_path, "a") as f:
            for step in self.data:
                f.write(json.dumps(step) + "\n")
        self.data = []

<img width="1076" height="446" alt="image" src="https://github.com/user-attachments/assets/b5814401-05e5-45ce-841c-9fafcb018c7a" />

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
    #我們只保留跟「移動」一樣多的「不動」，其他的 NONE 全部丟掉
    #這樣 AI 就不會覺得「不動」是常態了
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

<img width="1076" height="286" alt="image" src="https://github.com/user-attachments/assets/1bdb5998-9b76-41b3-b51b-03c67ed9c828" />

    class MLPlay:
        def __init__(self, ai_name, *args, **kwargs):
            print(f"[{ai_name}] 學生 AI 啟動！")
              self.side = ai_name
        
            dir_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(dir_path, "my_rf_model.pickle")
        
            if not os.path.exists(model_path):
                print(f"[Model] 錯誤：找不到模型檔案 {model_path}")
                self.model = None
            else:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                print("[Model] 模型讀取成功！")
<img width="1076" height="362" alt="image" src="https://github.com/user-attachments/assets/c37eee93-78e0-4e8c-b99a-eb4f8b82650c" />

    def update(self, scene_info, *args, **kwargs):
        if scene_info["status"] != "GAME_ALIVE" or not self.model:
            return "RESET"

        # 1. 取得原始資料
        ball_x = scene_info["ball"][0]
        ball_y = scene_info["ball"][1]
        speed_x = scene_info["ball_speed"][0]
        speed_y = scene_info["ball_speed"][1]
        
        if self.side == "1P":
            paddle_x = scene_info["platform_1P"][0]
        else:
            paddle_x = scene_info["platform_2P"][0]

        # 🔥【鍵關修改】如果是 2P，就使用「鏡像術」騙模型
        # 因為模型只看過 1P 的視角，所以我們要偽造數據
        if self.side == "2P":
            # 把 Y 座標上下顛倒 (假設視窗高度是 500)
            ball_y = 500 - ball_y
            # 把 Y 速度反向 (負變正，正變負)
            speed_y = -speed_y
            # X 座標不用變，因為左右對大家來說都是一樣的

        # 2. 把處理過的資料餵給模型
        feature = np.array([[ball_x, ball_y, speed_x, speed_y, paddle_x]])
        
        # 3. 預測
        action = self.model.predict(feature)
        
        return action[0]

    def reset(self):
        print(f"[{self.side}] 這一局結束。")
