class GameConfig:
    SCREEN_WIDTH = 1280
    SCREEN_HEIGHT = 720
    TARGET_FPS = 60
    MIN_FPS = 24
    
    MAX_HEALTH = 100
    ATTACK_DAMAGE = 15
    DEFENSE_BLOCK = 10
    
    GESTURE_DETECTION_CONFIDENCE = 0.6
    GESTURE_HOLD_FRAMES = 5
    
    COLORS = {
        'player_health': (0, 255, 0),
        'ai_health': (0, 0, 255),
        'text': (255, 255, 255),
        'attack_effect': (0, 165, 255),
        'defense_effect': (255, 255, 0),
        'damage_effect': (0, 0, 255),
    }


class PerformanceConfig:
    USE_MULTITHREAD_CAPTURE = True
    CAPTURE_BUFFER_SIZE = 1
    CAPTURE_FPS = 60
    
    DETECT_EVERY_N_FRAMES = 2
    INFERENCE_WIDTH = 320
    INFERENCE_HEIGHT = 240
    
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.4
    
    INITIAL_BUFFER_SIZE = 5
    MIN_BUFFER_SIZE = 3
    MAX_BUFFER_SIZE = 8
    
    ENABLE_PERFORMANCE_MONITOR = True
    SHOW_DEBUG_INFO = False
    SHOW_FPS = True


class CameraConfig:
    PREFER_CAMERA_BACKEND = 'dshow'
    PREFER_FOURCC = 'MJPG'
    PREFER_WIDTH = 1280
    PREFER_HEIGHT = 720
    PREFER_FPS = 60

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
