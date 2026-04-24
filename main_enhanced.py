import cv2
import time
import numpy as np
import threading

from config import (
    GameConfig, GestureConfig, PerformanceConfig, 
    CameraConfig, PreprocessingConfig
)
from video_capture_thread import VideoCaptureThread, PerformanceMonitor
from optimized_gesture import OptimizedHandDetector, AdaptiveGestureBuffer
from image_preprocessing import ImagePreprocessor, SkinDetector, AdvancedGestureClassifier
from game_logic import BattleSystem
from ui_renderer import UIRenderer


class EnhancedGestureBattleGame:
    def __init__(self, 
                 camera_id=0, 
                 use_multithread=True,
                 enable_preprocessing=True,
                 enable_skin_detection=True):
        
        self.camera_id = camera_id
        self.use_multithread = use_multithread
        self.enable_preprocessing = enable_preprocessing
        self.enable_skin_detection = enable_skin_detection
        
        self.video_thread = None
        self.cap = None
        
        self.hand_detector = None
        self.preprocessor = None
        self.skin_detector = None
        self.advanced_classifier = None
        self.gesture_buffer = None
        
        self.battle_system = None
        self.ui_renderer = None
        self.perf_monitor = None
        
        self.running = False
        self.frame_count = 0
        
        self.current_gesture = None
        self.gesture_confidence = 0.0
        self.last_skin_ratio = 0.0
        
        self.hand_roi = None
        self.preprocessed_frame = None
        self.skin_mask = None
        
        self.show_preprocessing = False
        self.show_skin_mask = False
        
        self._init_components()
    
    def _init_components(self):
        print("=" * 60)
        print("正在初始化游戏组件（增强版）")
        print("=" * 60)
        
        print(f"[配置] 多线程捕获: {'启用' if self.use_multithread else '禁用'}")
        print(f"[配置] 图像预处理: {'启用' if self.enable_preprocessing else '禁用'}")
        print(f"[配置] 皮肤检测: {'启用' if self.enable_skin_detection else '禁用'}")
        print(f"[配置] 检测跳帧: 每 {PerformanceConfig.DETECT_EVERY_N_FRAMES} 帧")
        
        self.hand_detector = OptimizedHandDetector(
            detect_every_n_frames=PerformanceConfig.DETECT_EVERY_N_FRAMES,
            inference_width=PerformanceConfig.INFERENCE_WIDTH,
            inference_height=PerformanceConfig.INFERENCE_HEIGHT,
            min_detection_confidence=PerformanceConfig.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=PerformanceConfig.MIN_TRACKING_CONFIDENCE
        )
        print("[OK] 优化手势检测器初始化完成")
        
        if self.enable_preprocessing:
            self.preprocessor = ImagePreprocessor(
                blur_kernel=PreprocessingConfig.BLUR_KERNEL,
                clahe_clip_limit=PreprocessingConfig.CLAHE_CLIP_LIMIT,
                clahe_grid_size=PreprocessingConfig.CLAHE_GRID_SIZE,
                enable_preprocessing=True
            )
            print("[OK] 图像预处理器初始化完成")
        else:
            self.preprocessor = None
        
        if self.enable_skin_detection:
            self.skin_detector = SkinDetector(
                ycrcb_lower=PreprocessingConfig.YCRCB_LOWER,
                ycrcb_upper=PreprocessingConfig.YCRCB_UPPER,
                hsv_lower=PreprocessingConfig.HSV_LOWER,
                hsv_upper=PreprocessingConfig.HSV_UPPER,
                use_adaptive_threshold=PreprocessingConfig.USE_ADAPTIVE_THRESHOLD
            )
            
            self.advanced_classifier = AdvancedGestureClassifier(
                use_skin_detection=True,
                use_contour_features=PreprocessingConfig.ENABLE_CONTOUR_FEATURES,
                min_skin_ratio=PreprocessingConfig.MIN_SKIN_RATIO
            )
            print("[OK] 皮肤检测器和高级分类器初始化完成")
        else:
            self.skin_detector = None
            self.advanced_classifier = None
        
        self.gesture_buffer = AdaptiveGestureBuffer(
            initial_buffer_size=PerformanceConfig.INITIAL_BUFFER_SIZE,
            min_buffer_size=PerformanceConfig.MIN_BUFFER_SIZE,
            max_buffer_size=PerformanceConfig.MAX_BUFFER_SIZE
        )
        print("[OK] 自适应手势缓冲区初始化完成")
        
        self.battle_system = BattleSystem()
        print("[OK] 战斗系统初始化完成")
        
        self.ui_renderer = UIRenderer()
        print("[OK] UI 渲染器初始化完成")
        
        if PerformanceConfig.ENABLE_PERFORMANCE_MONITOR:
            self.perf_monitor = PerformanceMonitor()
            print("[OK] 性能监控器初始化完成")
        
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
        
        original_frame = frame.copy()
        
        processed_for_detection = frame
        if self.preprocessor and self.enable_preprocessing:
            processed_for_detection = self.preprocessor.preprocess(frame)
        
        processed_for_detection = self.hand_detector.find_hands(
            processed_for_detection, draw=True
        )
        lm_list = self.hand_detector.find_position(processed_for_detection)
        
        if self.perf_monitor:
            self.perf_monitor.record_process()
        
        self.hand_roi = None
        if lm_list:
            if self.preprocessor:
                self.hand_roi, roi_image = self.preprocessor.extract_hand_roi(
                    original_frame, lm_list, padding=PreprocessingConfig.ROI_PADDING
                )
            else:
                h, w = original_frame.shape[:2]
                x_coords = [lm[1] for lm in lm_list]
                y_coords = [lm[2] for lm in lm_list]
                padding = PreprocessingConfig.ROI_PADDING
                min_x = max(0, min(x_coords) - padding)
                max_x = min(w, max(x_coords) + padding)
                min_y = max(0, min(y_coords) - padding)
                max_y = min(h, max(y_coords) + padding)
                self.hand_roi = (min_x, min_y, max_x - min_x, max_y - min_y)
        
        if self.enable_skin_detection and self.advanced_classifier:
            if lm_list:
                hand_valid, skin_ratio = self.advanced_classifier.validate_hand_presence(
                    original_frame, lm_list, self.hand_roi
                )
                self.last_skin_ratio = skin_ratio
                
                if hand_valid:
                    self.current_gesture, self.gesture_confidence = self.advanced_classifier.classify(
                        lm_list, original_frame, self.hand_roi
                    )
                else:
                    self.current_gesture = None
                    self.gesture_confidence = 0.0
            else:
                self.current_gesture = None
                self.gesture_confidence = 0.0
        else:
            if lm_list:
                from optimized_gesture import OptimizedGestureClassifier
                simple_classifier = OptimizedGestureClassifier()
                self.current_gesture, self.gesture_confidence = simple_classifier.classify(lm_list)
            else:
                self.current_gesture = None
                self.gesture_confidence = 0.0
        
        if lm_list and self.enable_skin_detection and self.skin_detector and self.hand_roi:
            skin_result = self.skin_detector.detect(original_frame, self.hand_roi)
            self.skin_mask = skin_result['mask']
        
        stable_gesture = self.gesture_buffer.add_gesture(self.current_gesture)
        
        if lm_list:
            self.gesture_buffer.update_stability(True)
        else:
            self.gesture_buffer.update_stability(False)
        
        self._handle_battle(stable_gesture)
        
        battle_state = self.battle_system.get_battle_state()
        
        display_frame = processed_for_detection if self.enable_preprocessing else frame
        
        self.ui_renderer.draw_player_ui(display_frame, battle_state)
        self.ui_renderer.draw_effects(display_frame)
        self.ui_renderer.draw_instructions(display_frame)
        
        if PerformanceConfig.SHOW_FPS and self.perf_monitor:
            stats = self.perf_monitor.get_stats()
            fps_text = f"FPS: {stats.get('fps_avg_ms', 0) * 1000:.0f}"
            cv2.putText(display_frame, fps_text, (1180, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        self._draw_additional_info(display_frame)
        
        if self.show_preprocessing and self.preprocessor and self.preprocessor.debug_info:
            self._show_debug_windows(original_frame)
        
        if self.perf_monitor:
            self.perf_monitor.record_render()
        
        return display_frame
    
    def _draw_additional_info(self, frame):
        y_offset = 150
        x_offset = 10
        
        lines = []
        
        if self.current_gesture:
            gesture_name = GestureConfig.GESTURE_NAMES.get(self.current_gesture, self.current_gesture)
            lines.append(f"手势: {gesture_name} ({self.gesture_confidence:.0%})")
        
        if self.enable_skin_detection:
            lines.append(f"皮肤比例: {self.last_skin_ratio:.1%}")
        
        if self.hand_roi:
            x, y, w, h = self.hand_roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "ROI", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        if PerformanceConfig.SHOW_DEBUG_INFO:
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (x_offset, y_offset + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            hand_stats = self.hand_detector.get_stats()
            debug_lines = [
                f"检测计数: {hand_stats['detection_count']}",
                f"跳帧计数: {hand_stats['skip_count']}",
                f"平均检测: {hand_stats['avg_detection_time_ms']:.2f}ms"
            ]
            
            for i, line in enumerate(debug_lines):
                cv2.putText(frame, line, (x_offset, 250 + i * 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def _show_debug_windows(self, original_frame):
        if 'denoised' in self.preprocessor.debug_info:
            cv2.imshow('Denoised', self.preprocessor.debug_info['denoised'])
        
        if 'enhanced' in self.preprocessor.debug_info:
            cv2.imshow('Enhanced', self.preprocessor.debug_info['enhanced'])
        
        if self.skin_mask is not None and self.hand_roi:
            x, y, w, h = self.hand_roi
            mask_display = cv2.cvtColor(self.skin_mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow('Skin Mask', mask_display)
    
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
            
            cv2.namedWindow("手势对战游戏 [增强版]", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("手势对战游戏 [增强版]", 
                           GameConfig.SCREEN_WIDTH, GameConfig.SCREEN_HEIGHT)
            
            print("\n游戏开始！使用手势进行对战：")
            print("  - 握拳 = 攻击")
            print("  - 张开手掌 = 防御")
            print("  - 按 'Q' 退出游戏")
            print("  - 按 'R' 重新开始")
            print("  - 按 'D' 切换调试信息")
            print("  - 按 'P' 切换预处理显示")
            print("\n增强功能已启用：")
            print("  - 多线程视频捕获")
            print("  - 图像预处理（降噪、亮度均衡）")
            print("  - 皮肤颜色检测（过滤假阳性）")
            print("  - ROI 感兴趣区域裁剪")
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
                
                cv2.imshow("手势对战游戏 [增强版]", processed_frame)
                
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
                elif key == ord('p'):
                    self.show_preprocessing = not self.show_preprocessing
                    print(f"\n预处理窗口: {'开启' if self.show_preprocessing else '关闭'}")
        
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
            
            if self.preprocessor:
                stats = self.preprocessor.get_stats()
                if stats['preprocess_count'] > 0:
                    print(f"\n图像预处理统计:")
                    print(f"  处理次数: {stats['preprocess_count']}")
                    print(f"  平均处理时间: {stats['avg_preprocess_time_ms']:.2f}ms")
            
            if self.advanced_classifier:
                stats = self.advanced_classifier.get_stats()
                print(f"\n高级分类器统计:")
                print(f"  分类次数: {stats['classification_count']}")
            
            cv2.destroyAllWindows()
            self.release_camera()
            print("游戏已结束")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='手势对战游戏 - 增强版（抗干扰优化）')
    parser.add_argument('--camera', type=int, default=0, help='摄像头设备ID (默认: 0)')
    parser.add_argument('--width', type=int, default=1280, help='窗口宽度 (默认: 1280)')
    parser.add_argument('--height', type=int, default=720, help='窗口高度 (默认: 720)')
    parser.add_argument('--no-multithread', action='store_true', help='禁用多线程捕获')
    parser.add_argument('--no-preprocessing', action='store_true', help='禁用图像预处理')
    parser.add_argument('--no-skin-detection', action='store_true', help='禁用皮肤检测')
    parser.add_argument('--debug', action='store_true', help='显示调试信息')
    parser.add_argument('--detect-frames', type=int, default=2, help='每N帧检测一次 (默认: 2)')
    parser.add_argument('--min-skin-ratio', type=float, default=0.1, help='最小皮肤比例阈值 (默认: 0.1)')
    
    args = parser.parse_args()
    
    GameConfig.SCREEN_WIDTH = args.width
    GameConfig.SCREEN_HEIGHT = args.height
    
    PerformanceConfig.DETECT_EVERY_N_FRAMES = args.detect_frames
    PreprocessingConfig.MIN_SKIN_RATIO = args.min_skin_ratio
    
    if args.debug:
        PerformanceConfig.SHOW_DEBUG_INFO = True
        print("[调试模式已启用]")
    
    use_multithread = not args.no_multithread
    enable_preprocessing = not args.no_preprocessing
    enable_skin_detection = not args.no_skin_detection
    
    print("\n" + "=" * 60)
    print("增强版手势对战游戏配置")
    print("=" * 60)
    print(f"多线程捕获:     {'启用' if use_multithread else '禁用'}")
    print(f"图像预处理:      {'启用' if enable_preprocessing else '禁用'}")
    print(f"皮肤检测:        {'启用' if enable_skin_detection else '禁用'}")
    print(f"检测跳帧:        每 {args.detect_frames} 帧")
    print(f"最小皮肤比例:    {args.min_skin_ratio:.0%}")
    print("=" * 60)
    
    game = EnhancedGestureBattleGame(
        camera_id=args.camera,
        use_multithread=use_multithread,
        enable_preprocessing=enable_preprocessing,
        enable_skin_detection=enable_skin_detection
    )
    game.run()


if __name__ == '__main__':
    main()
