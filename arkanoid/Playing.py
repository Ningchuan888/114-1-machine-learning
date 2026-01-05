import os, pickle, numpy as np

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