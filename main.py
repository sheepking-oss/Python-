import cv2
import time
import numpy as np

from config import GameConfig, GestureConfig
from gesture_recognition import HandDetector, GestureClassifier, GestureBuffer
from game_logic import BattleSystem
from ui_renderer import UIRenderer

class GestureBattleGame:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        
        self.hand_detector = HandDetector()
        self.gesture_classifier = GestureClassifier()
        self.gesture_buffer = GestureBuffer(buffer_size=GameConfig.GESTURE_HOLD_FRAMES)
        
        self.battle_system = BattleSystem()
        self.ui_renderer = UIRenderer()
        
        self.running = False
        self.last_frame_time = 0
        self.fps = 0
        self.fps_history = []
        
    def init_camera(self):
        print(f"正在初始化摄像头 (ID: {self.camera_id})...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise Exception(f"无法打开摄像头 ID: {self.camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, GameConfig.SCREEN_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, GameConfig.SCREEN_HEIGHT)
        
        print(f"摄像头初始化成功！分辨率: {GameConfig.SCREEN_WIDTH}x{GameConfig.SCREEN_HEIGHT}")
        
    def release_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            cv2.destroyAllWindows()
            print("摄像头已释放")
    
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        
        frame = self.hand_detector.find_hands(frame, draw=True)
        lm_list = self.hand_detector.find_position(frame)
        
        current_gesture, confidence = self.gesture_classifier.classify(lm_list)
        
        stable_gesture = self.gesture_buffer.add_gesture(current_gesture)
        
        self._handle_battle(stable_gesture)
        
        battle_state = self.battle_system.get_battle_state()
        
        self.ui_renderer.draw_player_ui(frame, battle_state)
        self.ui_renderer.draw_effects(frame)
        self.ui_renderer.draw_instructions(frame)
        self.ui_renderer.draw_fps(frame, self.fps)
        
        if current_gesture:
            gesture_name = GestureConfig.GESTURE_NAMES.get(current_gesture, current_gesture)
            cv2.putText(frame, f"当前手势: {gesture_name} ({confidence:.0%})", 
                        (500, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def _handle_battle(self, stable_gesture):
        if not stable_gesture:
            return
        
        game_over = self.battle_system.check_game_over()
        if game_over:
            return
        
        if self.battle_system.can_attack():
            result = self.battle_system.execute_round(stable_gesture)
            
            if result:
                if result['player_action'] == 'attack':
                    self.ui_renderer.add_attack_effect(is_player=True)
                else:
                    self.ui_renderer.add_defense_effect(is_player=True)
                
                if result['ai_action'] == 'attack':
                    self.ui_renderer.add_attack_effect(is_player=False)
                else:
                    self.ui_renderer.add_defense_effect(is_player=False)
                
                if result['player_damage'] > 0:
                    self.ui_renderer.add_damage_effect(is_player=True, damage=result['player_damage'])
                if result['ai_damage'] > 0:
                    self.ui_renderer.add_damage_effect(is_player=False, damage=result['ai_damage'])
    
    def _calculate_fps(self):
        current_time = time.time()
        if self.last_frame_time > 0:
            frame_time = current_time - self.last_frame_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            self.fps_history.append(current_fps)
            
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            
            self.fps = sum(self.fps_history) / len(self.fps_history)
        
        self.last_frame_time = current_time
    
    def run(self):
        try:
            self.init_camera()
            self.battle_system.start_battle()
            self.running = True
            
            cv2.namedWindow("手势对战游戏", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("手势对战游戏", GameConfig.SCREEN_WIDTH, GameConfig.SCREEN_HEIGHT)
            
            print("游戏开始！使用手势进行对战：")
            print("  - 握拳 = 攻击")
            print("  - 张开手掌 = 防御")
            print("  - 按 'Q' 退出游戏")
            print("  - 按 'R' 重新开始")
            
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("无法读取摄像头帧")
                    break
                
                frame = cv2.resize(frame, (GameConfig.SCREEN_WIDTH, GameConfig.SCREEN_HEIGHT))
                
                self._calculate_fps()
                
                processed_frame = self.process_frame(frame)
                
                cv2.imshow("手势对战游戏", processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    print("用户请求退出...")
                    self.running = False
                elif key == ord('r'):
                    print("重新开始游戏...")
                    self.battle_system.start_battle()
                    self.ui_renderer.effects = []
                    self.gesture_buffer.clear()
        
        except Exception as e:
            print(f"游戏运行出错: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.release_camera()
            print("游戏已结束")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='手势对战游戏 - 基于计算机视觉的互动对战小游戏')
    parser.add_argument('--camera', type=int, default=0, help='摄像头设备ID (默认: 0)')
    parser.add_argument('--width', type=int, default=1280, help='窗口宽度 (默认: 1280)')
    parser.add_argument('--height', type=int, default=720, help='窗口高度 (默认: 720)')
    
    args = parser.parse_args()
    
    GameConfig.SCREEN_WIDTH = args.width
    GameConfig.SCREEN_HEIGHT = args.height
    
    game = GestureBattleGame(camera_id=args.camera)
    game.run()

if __name__ == '__main__':
    main()
