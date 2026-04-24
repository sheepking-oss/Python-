import sys
import cv2
import mediapipe as mp
import numpy as np

print('OpenCV版本:', cv2.__version__)
print('MediaPipe版本:', mp.__version__)
print('NumPy版本:', np.__version__)

print('\n测试模块导入...')
from config import GameConfig, GestureConfig
from gesture_recognition import HandDetector, GestureClassifier, GestureBuffer
from game_logic import Player, AIOpponent, BattleSystem
from ui_renderer import UIRenderer
from main import GestureBattleGame
print('✓ 所有模块导入成功！')

print('\n测试组件初始化...')
hand_detector = HandDetector()
print('✓ HandDetector 初始化成功')

gesture_classifier = GestureClassifier()
print('✓ GestureClassifier 初始化成功')

gesture_buffer = GestureBuffer()
print('✓ GestureBuffer 初始化成功')

battle_system = BattleSystem()
print('✓ BattleSystem 初始化成功')

ui_renderer = UIRenderer()
print('✓ UIRenderer 初始化成功')

print('\n测试战斗系统...')
battle_system.start_battle()
state = battle_system.get_battle_state()
print(f'游戏状态: 玩家血量={state["player"]["health"]}, AI血量={state["ai"]["health"]}')
print('✓ 战斗系统测试成功！')

print('\n' + '='*50)
print('所有测试通过！游戏环境已就绪。')
print('='*50)
print('\n运行游戏命令:')
print('  python main.py')
print('\n游戏控制:')
print('  - 握拳 = 攻击')
print('  - 张开手掌 = 防御')
print('  - 按 Q 退出')
print('  - 按 R 重新开始')
