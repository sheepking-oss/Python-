import cv2
import time
import numpy as np
import threading

from config import GameConfig, GestureConfig, PerformanceConfig, CameraConfig
from video_capture_thread import VideoCaptureThread, PerformanceMonitor
from optimized_gesture import OptimizedHandDetector, OptimizedGestureClassifier, AdaptiveGestureBuffer
from game_logic import BattleSystem
from ui_renderer import UIRenderer


class OptimizedGestureBattleGame:
    def __init__(self, camera_id=0, use_multithread=True):
        self.camera_id = camera_id
        self.use_multithread = use_multithread
        
        self.cap = None
        self.video_thread = None
        
        self.hand_detector = None
        self.gesture_classifier = None
        self.gesture_buffer = None
        
        self.battle_system = None
        self.ui_renderer = None
        self.perf_monitor = None
        
        self.running = False
        self.frame_count = 0
        
        self.last_gesture = None
        self.current_gesture = None
        self.gesture_confidence = 0.0
        
        self._init_components()
    
    def _init_components(self):
        print("=" * 60)
        print("正在初始化游戏组件（性能优化模式）")
        print("=" * 60)
        
        print(f"[配置] 多线程捕获: {'启用' if self.use_multithread else '禁用'}")
        print(f"[配置] 检测跳帧: 每 {PerformanceConfig.DETECT_EVERY_N_FRAMES} 帧")
        print(f"[配置] 推理分辨率: {PerformanceConfig.INFERENCE_WIDTH}x{PerformanceConfig.INFERENCE_HEIGHT}")
        
        self.hand_detector = OptimizedHandDetector(
            detect_every_n_frames=PerformanceConfig.DETECT_EVERY_N_FRAMES,
            inference_width=PerformanceConfig.INFERENCE_WIDTH,
            inference_height=PerformanceConfig.INFERENCE_HEIGHT,
            min_detection_confidence=PerformanceConfig.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=PerformanceConfig.MIN_TRACKING_CONFIDENCE
        )
        print("[✓] 优化手势检测器初始化完成")
        
        self.gesture_classifier = OptimizedGestureClassifier()
        print("[✓] 优化手势分类器初始化完成")
        
        self.gesture_buffer = AdaptiveGestureBuffer(
            initial_buffer_size=PerformanceConfig.INITIAL_BUFFER_SIZE,
            min_buffer_size=PerformanceConfig.MIN_BUFFER_SIZE,
            max_buffer_size=PerformanceConfig.MAX_BUFFER_SIZE
        )
        print("[✓] 自适应手势缓冲区初始化完成")
        
        self.battle_system = BattleSystem()
        print("[✓] 战斗系统初始化完成")
        
        self.ui_renderer = UIRenderer()
        print("[✓] UI 渲染器初始化完成")
        
        if PerformanceConfig.ENABLE_PERFORMANCE_MONITOR:
            self.perf_monitor = PerformanceMonitor()
            print("[✓] 性能监控器初始化完成")
        
        print("=" * 60)
        print("所有组件初始化完成！")
        print("=" * 60)
    
    def init_camera(self):
        print(f"\n正在初始化摄像头 (ID: {self.camera_id})...")
        
        if self.use_multithread:
            self.video_thread = VideoCaptureThread(
                camera_id=self.camera_id,
                width=GameConfig.SCREEN_WIDTH,
                height=GameConfig.SCREEN_HEIGHT,
                fps=PerformanceConfig.CAPTURE_FPS,
                buffer_size=PerformanceConfig.CAPTURE_BUFFER_SIZE
            )
            self.video_thread.init_camera()
            self.video_thread.start()
        else:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise Exception(f"无法打开摄像头 ID: {self.camera_id}")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, GameConfig.SCREEN_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, GameConfig.SCREEN_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, PerformanceConfig.CAPTURE_FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print(f"摄像头初始化成功！分辨率: {GameConfig.SCREEN_WIDTH}x{GameConfig.SCREEN_HEIGHT}")
    
    def release_camera(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        
        cv2.destroyAllWindows()
        print("\n摄像头已释放")
    
    def get_frame(self):
        if self.use_multithread and self.video_thread:
            return self.video_thread.read()
        elif self.cap:
            return self.cap.read()
        return False, None
    
    def process_frame(self, frame):
        if self.perf_monitor:
            self.perf_monitor.record_capture()
        
        frame = cv2.flip(frame, 1)
        
        frame = self.hand_detector.find_hands(frame, draw=True)
        lm_list = self.hand_detector.find_position(frame)
        
        if self.perf_monitor:
            self.perf_monitor.record_process()
        
        self.current_gesture, self.gesture_confidence = self.gesture_classifier.classify(lm_list)
        
        self.gesture_buffer.update_stability(self.current_gesture is not None)
        
        stable_gesture = self.gesture_buffer.add_gesture(self.current_gesture)
        
        self._handle_battle(stable_gesture)
        
        battle_state = self.battle_system.get_battle_state()
        
        self.ui_renderer.draw_player_ui(frame, battle_state)
        self.ui_renderer.draw_effects(frame)
        self.ui_renderer.draw_instructions(frame)
        
        if PerformanceConfig.SHOW_FPS and self.perf_monitor:
            stats = self.perf_monitor.get_stats()
            fps_text = f"FPS: {stats.get('fps_avg_ms', 0) * 1000:.0f}"
            cv2.putText(frame, fps_text, (1180, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.current_gesture:
            gesture_name = GestureConfig.GESTURE_NAMES.get(self.current_gesture, self.current_gesture)
            cv2.putText(frame, f"当前手势: {gesture_name} ({self.gesture_confidence:.0%})", 
                        (500, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if PerformanceConfig.SHOW_DEBUG_INFO:
            self._draw_debug_info(frame)
        
        if self.perf_monitor:
            self.perf_monitor.record_render()
        
        return frame
    
    def _draw_debug_info(self, frame):
        y_offset = 400
        x_offset = 10
        
        hand_stats = self.hand_detector.get_stats()
        gesture_stats = self.gesture_classifier.get_stats()
        buffer_stats = self.gesture_buffer.get_stats()
        
        debug_lines = [
            "=== 调试信息 ===",
            f"检测计数: {hand_stats['detection_count']}",
            f"跳帧计数: {hand_stats['skip_count']}",
            f"平均检测: {hand_stats['avg_detection_time_ms']:.2f}ms",
            f"分类计数: {gesture_stats['classification_count']}",
            f"平均分类: {gesture_stats['avg_classification_time_ms']:.3f}ms",
            f"缓冲区大小: {buffer_stats['buffer_size']}/{buffer_stats['max_buffer_size']}",
            f"稳定手势: {buffer_stats['last_stable_gesture']}",
            f"自适应阈值: {buffer_stats['adaptive_threshold']:.1%}"
        ]
        
        if self.video_thread:
            video_stats = self.video_thread.get_stats()
            debug_lines.extend([
                "",
                "=== 视频捕获 ===",
                f"总帧数: {video_stats['frame_count']}",
                f"丢帧数: {video_stats['frame_drops']}",
                f"实际FPS: {video_stats['actual_fps']:.1f}",
                f"平均读取: {video_stats['avg_read_time_ms']:.2f}ms"
            ])
        
        for i, line in enumerate(debug_lines):
            cv2.putText(frame, line, (x_offset, y_offset + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
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
    
    def run(self):
        try:
            self.init_camera()
            self.battle_system.start_battle()
            self.running = True
            
            cv2.namedWindow("手势对战游戏 [性能优化版]", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("手势对战游戏 [性能优化版]", 
                           GameConfig.SCREEN_WIDTH, GameConfig.SCREEN_HEIGHT)
            
            print("\n游戏开始！使用手势进行对战：")
            print("  - 握拳 = 攻击")
            print("  - 张开手掌 = 防御")
            print("  - 按 'Q' 退出游戏")
            print("  - 按 'R' 重新开始")
            print("  - 按 'D' 切换调试信息")
            print("\n性能优化已启用：")
            print("  - 多线程视频捕获")
            print("  - 跳帧检测（每2帧检测一次）")
            print("  - 低分辨率推理（320x240）")
            print("  - 自适应手势缓冲区")
            print("-" * 60)
            
            frame_skip = 0
            max_frame_skip = 5
            
            while self.running:
                if self.perf_monitor:
                    self.perf_monitor.start_frame()
                
                ret, frame = self.get_frame()
                
                if not ret or frame is None:
                    frame_skip += 1
                    if frame_skip >= max_frame_skip:
                        print("警告: 连续无法读取帧")
                        frame_skip = 0
                    time.sleep(0.001)
                    continue
                
                frame_skip = 0
                self.frame_count += 1
                
                frame = cv2.resize(frame, (GameConfig.SCREEN_WIDTH, GameConfig.SCREEN_HEIGHT))
                
                processed_frame = self.process_frame(frame)
                
                cv2.imshow("手势对战游戏 [性能优化版]", processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    print("\n用户请求退出...")
                    self.running = False
                elif key == ord('r'):
                    print("\n重新开始游戏...")
                    self.battle_system.start_battle()
                    self.ui_renderer.effects = []
                    self.gesture_buffer.clear()
                elif key == ord('d'):
                    PerformanceConfig.SHOW_DEBUG_INFO = not PerformanceConfig.SHOW_DEBUG_INFO
                    print(f"\n调试信息: {'开启' if PerformanceConfig.SHOW_DEBUG_INFO else '关闭'}")
        
        except Exception as e:
            print(f"\n游戏运行出错: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if self.perf_monitor:
                print("\n" + self.perf_monitor.format_stats())
            
            if self.hand_detector:
                stats = self.hand_detector.get_stats()
                print(f"\n手势检测统计:")
                print(f"  检测次数: {stats['detection_count']}")
                print(f"  跳帧次数: {stats['skip_count']}")
                print(f"  平均检测时间: {stats['avg_detection_time_ms']:.2f}ms")
            
            self.release_camera()
            print("游戏已结束")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='手势对战游戏 - 性能优化版')
    parser.add_argument('--camera', type=int, default=0, help='摄像头设备ID (默认: 0)')
    parser.add_argument('--width', type=int, default=1280, help='窗口宽度 (默认: 1280)')
    parser.add_argument('--height', type=int, default=720, help='窗口高度 (默认: 720)')
    parser.add_argument('--no-multithread', action='store_true', help='禁用多线程捕获')
    parser.add_argument('--debug', action='store_true', help='显示调试信息')
    parser.add_argument('--detect-frames', type=int, default=2, help='每N帧检测一次 (默认: 2)')
    parser.add_argument('--inference-width', type=int, default=320, help='推理宽度 (默认: 320)')
    parser.add_argument('--inference-height', type=int, default=240, help='推理高度 (默认: 240)')
    
    args = parser.parse_args()
    
    GameConfig.SCREEN_WIDTH = args.width
    GameConfig.SCREEN_HEIGHT = args.height
    
    PerformanceConfig.DETECT_EVERY_N_FRAMES = args.detect_frames
    PerformanceConfig.INFERENCE_WIDTH = args.inference_width
    PerformanceConfig.INFERENCE_HEIGHT = args.inference_height
    
    if args.debug:
        PerformanceConfig.SHOW_DEBUG_INFO = True
        print("[调试模式已启用]")
    
    use_multithread = not args.no_multithread
    
    game = OptimizedGestureBattleGame(
        camera_id=args.camera,
        use_multithread=use_multithread
    )
    game.run()


if __name__ == '__main__':
    main()
