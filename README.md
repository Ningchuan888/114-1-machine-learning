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

# 打磚塊
## 架構
<img width="1798" height="827" alt="image" src="https://github.com/user-attachments/assets/0c5bef01-9a59-438d-8727-37bcabd6385c" />

## 設計
<img width="486" height="827" alt="image" src="https://github.com/user-attachments/assets/c1709992-8bf9-4232-a922-59e2f1095f12" />
<img width="1128" height="885" alt="image" src="https://github.com/user-attachments/assets/9e08e58c-bb35-44e3-ad64-25e7993068a0" />

## API
### 專案 : 資料搜集階段 (ml_play.py)
<img width="936" height="826" alt="image" src="https://github.com/user-attachments/assets/e42a61bf-bafb-4782-80ff-f718f03c9856" />

### 專案 : 模型訓練階段 (train_speed.py)
<img width="1270" height="569" alt="image" src="https://github.com/user-attachments/assets/aedae958-c735-406a-ade2-92a7c26e559c" />

### 專案 : 實際遊玩階段 (Playing.py)
<img width="1174" height="827" alt="image" src="https://github.com/user-attachments/assets/87c71649-6909-4ab9-8435-a725d60d8cbd" />

## API&程式碼
<img width="1280" height="621" alt="image" src="https://github.com/user-attachments/assets/6ae7e136-4550-462b-8f23-fecad8d10524" />

    def update(self, scene_info, *args, **kwargs):
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"

        if not scene_info["ball_served"]:
            self.ball_served = True
            self.previous_ball = scene_info["ball"]
            return "SERVE_TO_LEFT"

        # 1. 取得資訊
        ball_x = scene_info["ball"][0] + 2.5
        ball_y = scene_info["ball"][1] + 2.5
        platform_x, self.platform_y = scene_info["platform"]
        bricks = scene_info["bricks"] 
        
        vx = ball_x - (self.previous_ball[0] + 2.5)
        vy = ball_y - (self.previous_ball[1] + 2.5)

        # 2. [物理預測]
        if vy > 0:
            steps = (self.platform_y - ball_y) / vy 
            raw_pred_x = ball_x + (vx * steps)
            while raw_pred_x < 0 or raw_pred_x > 200:
                if raw_pred_x < 0: raw_pred_x = -raw_pred_x
                elif raw_pred_x > 200: raw_pred_x = 400 - raw_pred_x
        else:
            raw_pred_x = ball_x
            # 球往上飛，準備換下一個雨刷角度
            if ball_y > 350: 
                 pass 

        # 3. [策略核心]
        if vy > 0: # 球往下掉
            
            # --- [A. 絕對防守] ---
            if ball_y > 300 or abs(vy) > 10:
                self.locked_offset = 0
            
            else:
                num_bricks = len(bricks)
                
                # --- [B. 殘局動態切球 (Active Wiper)] ---
                if 0 < num_bricks <= 3:
                    
                    # 隨機切換角度
                    if random.random() < 0.08: 
                        self.wiper_index = (self.wiper_index + 1) % len(self.wiper_angles)
                        print(f"極限切球中... 剩 {num_bricks} 磚 | 角度: {self.wiper_angles[self.wiper_index]}")
                    
                    angle = self.wiper_angles[self.wiper_index]
                    self.locked_offset = angle

                # --- [C. 正常導引] ---
                elif num_bricks > 3:
                    if self.target_brick is None or self.target_brick not in bricks:
                        sorted_bricks = sorted(bricks, key=lambda b: b[1], reverse=True)
                        self.target_brick = sorted_bricks[0]

                    target_x = self.target_brick[0]
                    dx = target_x - raw_pred_x 
                    calculated_offset = dx * 0.35
                    
                    if calculated_offset > 17: calculated_offset = 17
                    elif calculated_offset < -17: calculated_offset = -17
                    
                    self.locked_offset = int(calculated_offset)
                    
                    if abs(vx) < 1.0:
                        self.locked_offset = random.choice([15, -15])
                else:
                    self.locked_offset = 0

            # 計算最終板子目標
            self.pred_x = raw_pred_x - self.locked_offset
        
        else: # 球往上飛
            self.pred_x = ball_x

        # 4. 移動動作
        platform_center = platform_x + 20
        action = 2 
        
        # --- [關鍵修正：動態切球邏輯] ---
        # 如果正在殘局雨刷模式，我們要讓板子「動起來」
        # 透過故意製造誤差，讓板子為了追球而產生速度
        
        real_target = self.pred_x
        
        # 如果要切右邊 (Offset > 0)，我們故意瞄準球的右邊一點點
        # 這樣板子會為了追球而往左跑，產生左旋？不，反過來。
        # 總之，為了製造切球效果，我們讓板子保持移動狀態。
        
        if platform_center < real_target - 2:
            action = 1 
        elif platform_center > real_target + 2:
            action = 0 
        else:
            # 如果已經對準了，但在殘局模式下，我們不要停！
            # 強制微動，保持板子有速度
            if len(bricks) <= 3:
                action = random.choice([0, 1]) # 原地抖動，保持活性
            else:
                action = 2

        # 5. 存入記憶體
        self.data.append([ball_x, ball_y, platform_x, vx, vy])
        self.target.append(action)
        self.previous_ball = scene_info["ball"]

        if action == 0: return "MOVE_LEFT"
        elif action == 1: return "MOVE_RIGHT"
        else: return "NONE"
<img width="1278" height="397" alt="image" src="https://github.com/user-attachments/assets/7d270d98-a982-4c65-946e-9949da3c198c" />

    def reset(self):
        print(f"Game Over. Saving {len(self.data)} rows...")
        with open(self.data_path, "wb") as f:
            pickle.dump(self.data, f)
        with open(self.target_path, "wb") as f:
            pickle.dump(self.target, f)
        self.ball_served = False
<img width="1272" height="515" alt="image" src="https://github.com/user-attachments/assets/e9ce4a1c-51c0-483d-b37c-ae9aa1dba9dd" />

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
<img width="1270" height="343" alt="image" src="https://github.com/user-attachments/assets/0c8ef570-640b-4120-9fbf-f67871afd536" />

     class MLPlay:
     　　def __init__(self, ai_name, *args, **kwargs):
            self.ball_served = False
            self.previous_ball = (0, 0)
        
            # 1. 載入模型
            path = os.path.dirname(__file__)
            model_path = os.path.join(path, "rf_model.pickle")
        
            if os.path.exists(model_path):
            　　with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
               print(f" 成功載入隨機森林模型: rf_model.pickle")
           　else:
                 print("找不到模型檔！請先執行 train_rf.py")
                 self.model = None
<img width="1270" height="451" alt="image" src="https://github.com/user-attachments/assets/836fa9f4-06a6-418f-b641-172d22e0b5fc" />

    def update(self, scene_info, *args, **kwargs):
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"

        if not scene_info["ball_served"]:
            self.ball_served = True
            self.previous_ball = scene_info["ball"]
            return "SERVE_TO_LEFT"

        # 2. 整理特徵 (Input)
        ball_x = scene_info["ball"][0] + 2.5
        ball_y = scene_info["ball"][1] + 2.5
        platform_x = scene_info["platform"][0]
        vx = ball_x - (self.previous_ball[0] + 2.5)
        vy = ball_y - (self.previous_ball[1] + 2.5)
        self.previous_ball = (scene_info["ball"][0], scene_info["ball"][1])

        # 3. 模型預測 (Prediction)
        # 只有當球往下掉 (vy > 0) 時才問模型，節省資源避免 Delay
        # 球往上飛時，簡單跟隨 X 軸即可
        if self.model and vy > 0:
            input_data = np.array([[ball_x, ball_y, platform_x, vx, vy]])
            action_code = self.model.predict(input_data)[0]
            return ["MOVE_LEFT", "MOVE_RIGHT", "NONE"][action_code]
        else:
            # 球往上飛時的簡單省電邏輯
            if platform_x + 20 < ball_x - 2: return "MOVE_RIGHT"
            elif platform_x + 20 > ball_x + 2: return "MOVE_LEFT"
            return "NONE"

    def reset(self):
        self.ball_served = False

# 分工表
<img width="491" height="170" alt="image" src="https://github.com/user-attachments/assets/37ae6e88-67b5-4b13-8edd-f3042fdf6cf0" />

# 參考資料
PAIA遊戲：https://app.paia-arena.com/game/1/code/20128/play?difficulty=EASY&level=1&levelfile

