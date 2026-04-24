import sys
import os

def test_imports():
    print("=" * 50)
    print("测试模块导入...")
    print("=" * 50)
    
    try:
        import cv2
        print(f"✓ OpenCV 导入成功，版本: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV 导入失败: {e}")
        return False
    
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe 导入成功，版本: {mp.__version__}")
    except ImportError as e:
        print(f"✗ MediaPipe 导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy 导入成功，版本: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy 导入失败: {e}")
        return False
    
    try:
        from config import GameConfig, GestureConfig
        print("✓ config 模块导入成功")
    except ImportError as e:
        print(f"✗ config 模块导入失败: {e}")
        return False
    
    try:
        from gesture_recognition import HandDetector, GestureClassifier, GestureBuffer
        print("✓ gesture_recognition 模块导入成功")
    except ImportError as e:
        print(f"✗ gesture_recognition 模块导入失败: {e}")
        return False
    
    try:
        from game_logic import Player, AIOpponent, BattleSystem
        print("✓ game_logic 模块导入成功")
    except ImportError as e:
        print(f"✗ game_logic 模块导入失败: {e}")
        return False
    
    try:
        from ui_renderer import UIRenderer
        print("✓ ui_renderer 模块导入成功")
    except ImportError as e:
        print(f"✗ ui_renderer 模块导入失败: {e}")
        return False
    
    try:
        from main import GestureBattleGame
        print("✓ main 模块导入成功")
    except ImportError as e:
        print(f"✗ main 模块导入失败: {e}")
        return False
    
    return True

def test_components():
    print("\n" + "=" * 50)
    print("测试组件初始化...")
    print("=" * 50)
    
    try:
        from gesture_recognition import HandDetector, GestureClassifier, GestureBuffer
        
        hand_detector = HandDetector()
        print("✓ HandDetector 初始化成功")
        
        gesture_classifier = GestureClassifier()
        print("✓ GestureClassifier 初始化成功")
        
        gesture_buffer = GestureBuffer()
        print("✓ GestureBuffer 初始化成功")
        
    except Exception as e:
        print(f"✗ 手势识别组件初始化失败: {e}")
        return False
    
    try:
        from game_logic import BattleSystem
        
        battle_system = BattleSystem()
        battle_system.start_battle()
        print("✓ BattleSystem 初始化成功")
        
        state = battle_system.get_battle_state()
        print(f"  - 玩家血量: {state['player']['health']}")
        print(f"  - AI 血量: {state['ai']['health']}")
        
    except Exception as e:
        print(f"✗ 战斗系统初始化失败: {e}")
        return False
    
    try:
        from ui_renderer import UIRenderer
        
        ui_renderer = UIRenderer()
        print("✓ UIRenderer 初始化成功")
        
    except Exception as e:
        print(f"✗ UI 渲染器初始化失败: {e}")
        return False
    
    return True

def test_camera():
    print("\n" + "=" * 50)
    print("测试摄像头访问...")
    print("=" * 50)
    
    import cv2
    
    print("正在尝试打开摄像头 (ID: 0)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ 无法打开摄像头，请检查摄像头是否连接或被其他程序占用")
        print("提示: 可以尝试使用 --camera 参数指定其他摄像头 ID")
        return False
    
    ret, frame = cap.read()
    
    if ret:
        print(f"✓ 摄像头访问成功！帧尺寸: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("✗ 无法读取摄像头帧")
        cap.release()
        return False
    
    cap.release()
    return True

def main():
    print("手势对战游戏 - 环境测试")
    print("=" * 50)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_components():
        all_passed = False
    
    if not test_camera():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ 所有测试通过！游戏环境已就绪。")
        print("\n运行游戏命令:")
        print("  python main.py")
        print("\n可选参数:")
        print("  --camera N  指定摄像头 ID (默认: 0)")
        print("  --width N   窗口宽度 (默认: 1280)")
        print("  --height N  窗口高度 (默认: 720)")
        print("\n游戏控制:")
        print("  - 握拳 = 攻击")
        print("  - 张开手掌 = 防御")
        print("  - 按 'Q' 退出")
        print("  - 按 'R' 重新开始")
    else:
        print("✗ 部分测试失败，请检查上述错误信息。")
    print("=" * 50)
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
