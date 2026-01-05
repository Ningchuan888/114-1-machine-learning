import sys
import os
import pickle
import numpy as np

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