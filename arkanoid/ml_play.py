"""
版本：動態切球雨刷 (Active Slicing Wiper)
針對：剩最後 1~2 塊磚的極限死角。
策略：
1. [動態切球] 擊球瞬間強制板子移動，製造極限反射角 (香蕉球)。
2. [雨刷輪替] 依然保留角度輪替，但加上了切球動作。
3. [極速運算] RAM Only，無 Delay。
"""
import os
import pickle
import numpy as np
import random

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        print("Mode: Active Slicing Wiper (Banana Shot)")
        self.ball_served = False
        self.previous_ball = (0, 0)
        self.pred_x = 100
        self.platform_y = 400
        
        # --- 變數 ---
        self.target_brick = None 
        self.locked_offset = 0   
        
        # 雨刷角度 (加大角度範圍)
        self.wiper_angles = [-20, -15, -10, -5, 0, 5, 10, 15, 20] 
        self.wiper_index = 0 

        # 資料庫
        path = os.path.dirname(__file__)
        self.data_path = os.path.join(path, "data.pickle")
        self.target_path = os.path.join(path, "target.pickle")
        
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "rb") as f:
                    self.data = pickle.load(f)
                with open(self.target_path, "rb") as f:
                    self.target = pickle.load(f)
                print(f"已載入: {len(self.data)} 筆資料")
            except:
                self.data = []
                self.target = []
        else:
            self.data = []
            self.target = []

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

    def reset(self):
        print(f"Game Over. Saving {len(self.data)} rows...")
        with open(self.data_path, "wb") as f:
            pickle.dump(self.data, f)
        with open(self.target_path, "wb") as f:
            pickle.dump(self.target, f)
        self.ball_served = False