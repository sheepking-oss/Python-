class GameConfig:
    SCREEN_WIDTH = 1280
    SCREEN_HEIGHT = 720
    FPS = 30
    
    MAX_HEALTH = 100
    ATTACK_DAMAGE = 15
    DEFENSE_BLOCK = 10
    
    GESTURE_DETECTION_CONFIDENCE = 0.7
    GESTURE_HOLD_FRAMES = 10
    
    COLORS = {
        'player_health': (0, 255, 0),
        'ai_health': (0, 0, 255),
        'text': (255, 255, 255),
        'attack_effect': (0, 165, 255),
        'defense_effect': (255, 255, 0),
        'damage_effect': (0, 0, 255),
    }

class GestureConfig:
    GESTURES = {
        0: 'rock',
        1: 'paper',
        2: 'scissors',
        3: 'open',
        4: 'fist'
    }
    
    GESTURE_NAMES = {
        'rock': '石头',
        'paper': '布',
        'scissors': '剪刀',
        'open': '防御',
        'fist': '攻击'
    }
    
    BATTLE_ACTIONS = {
        'rock': 'attack',
        'paper': 'defend',
        'scissors': 'attack',
        'open': 'defend',
        'fist': 'attack'
    }
