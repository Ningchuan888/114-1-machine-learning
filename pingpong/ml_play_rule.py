import os
import json

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.side = ai_name
        self.data = [] 
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        print(f"[{ai_name}] 老師程式 (精簡數據版) 啟動！")

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
            
# --- 移動邏輯 ---
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