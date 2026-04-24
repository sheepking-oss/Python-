import cv2
import numpy as np
import time
from config import GameConfig, GestureConfig

class UIRenderer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.thickness = 2
        self.effects = []
        
    def draw_health_bar(self, frame, x, y, width, height, current_health, max_health, color, label):
        cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
        
        health_percent = current_health / max_health
        bar_width = int(width * health_percent)
        
        if health_percent > 0.6:
            bar_color = color
        elif health_percent > 0.3:
            bar_color = (0, 255, 255)
        else:
            bar_color = (0, 0, 255)
        
        cv2.rectangle(frame, (x, y), (x + bar_width, y + height), bar_color, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
        
        text = f"{label}: {current_health}/{max_health}"
        text_size = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)[0]
        text_x = x + (width - text_size[0]) // 2
        text_y = y - 10
        
        cv2.putText(frame, text, (text_x, text_y), self.font, self.font_scale, 
                    GameConfig.COLORS['text'], self.thickness)
    
    def draw_player_ui(self, frame, battle_state):
        player = battle_state['player']
        ai = battle_state['ai']
        
        self.draw_health_bar(
            frame, 50, 80, 300, 30,
            player['health'], player['max_health'],
            GameConfig.COLORS['player_health'], player['name']
        )
        
        self.draw_health_bar(
            frame, 930, 80, 300, 30,
            ai['health'], ai['max_health'],
            GameConfig.COLORS['ai_health'], ai['name']
        )
        
        round_text = f"回合: {battle_state['round']}"
        cv2.putText(frame, round_text, (600, 50), self.font, 1.0,
                    GameConfig.COLORS['text'], 2)
        
        if not battle_state['can_attack']:
            cooldown = battle_state['cooldown_remaining']
            cooldown_text = f"冷却中: {cooldown:.1f}s"
            cv2.putText(frame, cooldown_text, (550, 100), self.font, 1.0,
                        (0, 165, 255), 2)
        
        self._draw_gesture_display(frame, player['gesture'], 200, 500, "玩家手势")
        self._draw_gesture_display(frame, ai['gesture'], 1000, 500, "AI 手势")
        
        self._draw_battle_log(frame, battle_state['battle_log'])
        
        game_over = battle_state['game_over']
        if game_over:
            self._draw_game_over(frame, game_over)
    
    def _draw_gesture_display(self, frame, gesture, x, y, label):
        if gesture:
            gesture_name = GestureConfig.GESTURE_NAMES.get(gesture, gesture)
            text = f"{label}: {gesture_name}"
        else:
            text = f"{label}: 等待输入"
        
        cv2.rectangle(frame, (x - 120, y - 30), (x + 120, y + 50), (0, 0, 0), -1)
        cv2.rectangle(frame, (x - 120, y - 30), (x + 120, y + 50), (255, 255, 255), 2)
        
        text_size = cv2.getTextSize(text, self.font, 0.7, 2)[0]
        text_x = x - text_size[0] // 2
        
        cv2.putText(frame, text, (text_x, y + 10), self.font, 0.7,
                    GameConfig.COLORS['text'], 2)
    
    def _draw_battle_log(self, frame, battle_log):
        y_offset = 150
        cv2.rectangle(frame, (400, y_offset), (880, y_offset + 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (400, y_offset), (880, y_offset + 180), (255, 255, 255), 2)
        
        cv2.putText(frame, "战斗日志", (420, y_offset + 25), self.font, 0.7,
                    (255, 255, 0), 2)
        
        for i, log in enumerate(battle_log[-5:]):
            cv2.putText(frame, log, (420, y_offset + 50 + i * 25), self.font, 0.5,
                        GameConfig.COLORS['text'], 1)
    
    def _draw_game_over(self, frame, result):
        overlay = frame.copy()
        cv2.rectangle(overlay, (300, 200), (980, 500), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        if result == 'player_win':
            text = "恭喜你获胜！"
            color = (0, 255, 0)
        elif result == 'ai_win':
            text = "AI 获胜了！"
            color = (0, 0, 255)
        else:
            text = "平局！"
            color = (255, 255, 0)
        
        text_size = cv2.getTextSize(text, self.font, 2.0, 4)[0]
        text_x = (GameConfig.SCREEN_WIDTH - text_size[0]) // 2
        text_y = 350
        
        cv2.putText(frame, text, (text_x, text_y), self.font, 2.0, color, 4)
        
        restart_text = "按 'R' 重新开始"
        restart_size = cv2.getTextSize(restart_text, self.font, 1.0, 2)[0]
        restart_x = (GameConfig.SCREEN_WIDTH - restart_size[0]) // 2
        
        cv2.putText(frame, restart_text, (restart_x, 420), self.font, 1.0,
                    GameConfig.COLORS['text'], 2)
    
    def add_attack_effect(self, is_player):
        effect = {
            'type': 'attack',
            'is_player': is_player,
            'start_time': time.time(),
            'duration': 0.5,
            'intensity': 1.0
        }
        self.effects.append(effect)
    
    def add_defense_effect(self, is_player):
        effect = {
            'type': 'defense',
            'is_player': is_player,
            'start_time': time.time(),
            'duration': 0.5,
            'intensity': 1.0
        }
        self.effects.append(effect)
    
    def add_damage_effect(self, is_player, damage):
        effect = {
            'type': 'damage',
            'is_player': is_player,
            'damage': damage,
            'start_time': time.time(),
            'duration': 1.0,
            'intensity': 1.0
        }
        self.effects.append(effect)
    
    def draw_effects(self, frame):
        current_time = time.time()
        active_effects = []
        
        for effect in self.effects:
            elapsed = current_time - effect['start_time']
            if elapsed < effect['duration']:
                progress = elapsed / effect['duration']
                self._draw_single_effect(frame, effect, progress)
                active_effects.append(effect)
        
        self.effects = active_effects
    
    def _draw_single_effect(self, frame, effect, progress):
        alpha = 1.0 - progress
        
        if effect['type'] == 'attack':
            x = 200 if effect['is_player'] else 1000
            y = 400
            color = GameConfig.COLORS['attack_effect']
            self._draw_pulse_effect(frame, x, y, color, alpha, progress)
            
        elif effect['type'] == 'defense':
            x = 200 if effect['is_player'] else 1000
            y = 400
            color = GameConfig.COLORS['defense_effect']
            self._draw_shield_effect(frame, x, y, color, alpha, progress)
            
        elif effect['type'] == 'damage':
            x = 200 if effect['is_player'] else 1000
            y = 350 - int(progress * 100)
            damage_text = f"-{effect['damage']}"
            self._draw_damage_text(frame, x, y, damage_text, alpha)
    
    def _draw_pulse_effect(self, frame, x, y, color, alpha, progress):
        radius = int(50 + progress * 100)
        thickness = max(1, int(10 * (1 - progress)))
        
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), radius, color, thickness)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def _draw_shield_effect(self, frame, x, y, color, alpha, progress):
        radius = int(80 * (1 - progress * 0.3))
        thickness = max(2, int(5 * (1 - progress)))
        
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), radius, color, thickness)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def _draw_damage_text(self, frame, x, y, text, alpha):
        overlay = frame.copy()
        cv2.putText(overlay, text, (x - 40, y), self.font, 1.5,
                    GameConfig.COLORS['damage_effect'], 3)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def draw_instructions(self, frame):
        instructions = [
            "游戏控制:",
            "握拳 = 攻击",
            "张开手掌 = 防御",
            "按 'Q' 退出",
            "按 'R' 重新开始"
        ]
        
        y_offset = 600
        x_offset = 20
        
        cv2.rectangle(frame, (x_offset, y_offset - 20), (250, y_offset + 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (x_offset, y_offset - 20), (250, y_offset + 120), (255, 255, 255), 1)
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (x_offset + 10, y_offset + i * 20),
                        self.font, 0.5, GameConfig.COLORS['text'], 1)
    
    def draw_fps(self, frame, fps):
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (1180, 30), self.font, 0.7,
                    (0, 255, 0), 2)
