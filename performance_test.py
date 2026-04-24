import cv2
import time
import numpy as np
import sys
import os

from collections import deque


class PerformanceBenchmark:
    def __init__(self):
        self.results = {}
        
    def benchmark_cv2_capture(self, camera_id=0, duration=5.0, width=1280, height=720):
        print("\n" + "=" * 60)
        print("基准测试 1: OpenCV 原生捕获 (单线程)")
        print("=" * 60)
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 ID: {camera_id}")
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        frame_times = []
        read_times = []
        frame_count = 0
        
        print(f"正在测试 {duration} 秒...")
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < duration:
            frame_start = time.perf_counter()
            
            ret, frame = cap.read()
            
            read_end = time.perf_counter()
            
            if ret:
                frame_count += 1
                read_times.append(read_end - frame_start)
                frame_times.append(time.perf_counter() - frame_start)
        
        cap.release()
        
        actual_duration = time.perf_counter() - start_time
        avg_fps = frame_count / actual_duration
        
        avg_read_time = sum(read_times) / len(read_times) if read_times else 0
        max_read_time = max(read_times) if read_times else 0
        min_read_time = min(read_times) if read_times else 0
        
        results = {
            'test_name': 'OpenCV 原生捕获',
            'frame_count': frame_count,
            'duration': actual_duration,
            'avg_fps': avg_fps,
            'avg_read_time_ms': avg_read_time * 1000,
            'max_read_time_ms': max_read_time * 1000,
            'min_read_time_ms': min_read_time * 1000,
            'latency_ms': avg_read_time * 1000
        }
        
        print(f"结果:")
        print(f"  总帧数: {frame_count}")
        print(f"  平均 FPS: {avg_fps:.1f}")
        print(f"  平均读取时间: {avg_read_time * 1000:.2f}ms")
        print(f"  读取时间范围: {min_read_time * 1000:.2f}ms - {max_read_time * 1000:.2f}ms")
        
        return results
    
    def benchmark_threaded_capture(self, camera_id=0, duration=5.0, width=1280, height=720):
        print("\n" + "=" * 60)
        print("基准测试 2: 多线程捕获")
        print("=" * 60)
        
        try:
            from video_capture_thread import VideoCaptureThread
        except ImportError as e:
            print(f"错误: 无法导入 VideoCaptureThread: {e}")
            return None
        
        video_thread = VideoCaptureThread(
            camera_id=camera_id,
            width=width,
            height=height,
            fps=60,
            buffer_size=1
        )
        
        try:
            video_thread.init_camera()
        except Exception as e:
            print(f"错误: 无法初始化摄像头: {e}")
            return None
        
        video_thread.start()
        time.sleep(0.5)
        
        frame_times = []
        frame_count = 0
        empty_count = 0
        
        print(f"正在测试 {duration} 秒...")
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < duration:
            frame_start = time.perf_counter()
            
            ret, frame = video_thread.read()
            
            if ret and frame is not None:
                frame_count += 1
                frame_times.append(time.perf_counter() - frame_start)
            else:
                empty_count += 1
                time.sleep(0.001)
        
        stats = video_thread.get_stats()
        video_thread.stop()
        
        actual_duration = time.perf_counter() - start_time
        avg_fps = frame_count / actual_duration
        
        avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
        
        results = {
            'test_name': '多线程捕获',
            'frame_count': frame_count,
            'empty_reads': empty_count,
            'duration': actual_duration,
            'avg_fps': avg_fps,
            'avg_frame_time_ms': avg_frame_time * 1000,
            'video_thread_fps': stats['actual_fps'],
            'video_thread_drops': stats['frame_drops'],
            'latency_ms': avg_frame_time * 1000
        }
        
        print(f"结果:")
        print(f"  总帧数: {frame_count}")
        print(f"  空读取: {empty_count}")
        print(f"  平均 FPS: {avg_fps:.1f}")
        print(f"  视频线程 FPS: {stats['actual_fps']:.1f}")
        print(f"  视频线程丢帧: {stats['frame_drops']}")
        print(f"  平均帧获取时间: {avg_frame_time * 1000:.2f}ms")
        
        return results
    
    def benchmark_mediapipe_detection(self, camera_id=0, duration=5.0, 
                                        detect_every_n=1, inference_scale=1.0):
        import mediapipe as mp
        
        print("\n" + "=" * 60)
        print(f"基准测试 3: MediaPipe 手势检测")
        print(f"  检测频率: 每 {detect_every_n} 帧")
        print(f"  推理缩放: {inference_scale:.1%}")
        print("=" * 60)
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4
        )
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 ID: {camera_id}")
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frame_count = 0
        detection_count = 0
        detection_times = []
        total_times = []
        
        print(f"正在测试 {duration} 秒...")
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < duration:
            total_start = time.perf_counter()
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            if frame_count % detect_every_n == 0:
                detection_start = time.perf_counter()
                
                h, w = frame.shape[:2]
                new_w = int(w * inference_scale)
                new_h = int(h * inference_scale)
                
                if inference_scale < 1.0:
                    small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    small_frame = frame
                
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                detection_count += 1
                detection_times.append(time.perf_counter() - detection_start)
            
            total_times.append(time.perf_counter() - total_start)
        
        cap.release()
        hands.close()
        
        actual_duration = time.perf_counter() - start_time
        avg_fps = frame_count / actual_duration
        
        avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
        max_detection_time = max(detection_times) if detection_times else 0
        
        avg_total_time = sum(total_times) / len(total_times) if total_times else 0
        
        results = {
            'test_name': f'MediaPipe检测 (N={detect_every_n}, scale={inference_scale})',
            'detect_every_n': detect_every_n,
            'inference_scale': inference_scale,
            'frame_count': frame_count,
            'detection_count': detection_count,
            'duration': actual_duration,
            'avg_fps': avg_fps,
            'avg_detection_time_ms': avg_detection_time * 1000,
            'max_detection_time_ms': max_detection_time * 1000,
            'avg_total_time_ms': avg_total_time * 1000,
            'latency_ms': avg_detection_time * 1000
        }
        
        print(f"结果:")
        print(f"  总帧数: {frame_count}")
        print(f"  检测次数: {detection_count}")
        print(f"  平均 FPS: {avg_fps:.1f}")
        print(f"  平均检测时间: {avg_detection_time * 1000:.2f}ms")
        print(f"  最大检测时间: {max_detection_time * 1000:.2f}ms")
        print(f"  平均总帧时间: {avg_total_time * 1000:.2f}ms")
        
        return results
    
    def run_all_benchmarks(self, camera_id=0, duration=3.0):
        print("\n" + "#" * 60)
        print("# 性能对比基准测试")
        print("#" * 60)
        
        all_results = []
        
        result1 = self.benchmark_cv2_capture(camera_id, duration)
        if result1:
            all_results.append(result1)
        
        result2 = self.benchmark_threaded_capture(camera_id, duration)
        if result2:
            all_results.append(result2)
        
        result3 = self.benchmark_mediapipe_detection(camera_id, duration, detect_every_n=1, inference_scale=1.0)
        if result3:
            all_results.append(result3)
        
        result4 = self.benchmark_mediapipe_detection(camera_id, duration, detect_every_n=2, inference_scale=0.5)
        if result4:
            all_results.append(result4)
        
        result5 = self.benchmark_mediapipe_detection(camera_id, duration, detect_every_n=3, inference_scale=0.25)
        if result5:
            all_results.append(result5)
        
        print("\n" + "#" * 60)
        print("# 综合对比结果")
        print("#" * 60)
        
        print(f"\n{'测试名称':<40} {'平均FPS':<10} {'延迟(ms)':<12} {'检测时间(ms)':<15}")
        print("-" * 80)
        
        for result in all_results:
            name = result.get('test_name', 'N/A')[:38]
            fps = f"{result.get('avg_fps', 0):.1f}"
            latency = f"{result.get('latency_ms', 0):.2f}"
            detect_time = f"{result.get('avg_detection_time_ms', 0):.2f}"
            
            print(f"{name:<40} {fps:<10} {latency:<12} {detect_time:<15}")
        
        print("\n" + "=" * 60)
        print("优化建议:")
        print("=" * 60)
        print("1. 使用多线程捕获可显著降低延迟并提高帧率稳定性")
        print("2. 跳帧检测 (每2-3帧检测一次) 可大幅降低 CPU 使用率")
        print("3. 降低推理分辨率 (320x240) 可显著减少推理时间")
        print("4. 组合使用以上优化可实现流畅的实时体验")
        print("=" * 60)
        
        return all_results


def quick_test():
    print("快速测试优化后的游戏组件...")
    
    try:
        from config import GameConfig, PerformanceConfig
        print(f"[✓] 配置加载成功")
        print(f"    - 目标帧率: {GameConfig.TARGET_FPS}")
        print(f"    - 检测跳帧: 每 {PerformanceConfig.DETECT_EVERY_N_FRAMES} 帧")
        print(f"    - 推理分辨率: {PerformanceConfig.INFERENCE_WIDTH}x{PerformanceConfig.INFERENCE_HEIGHT}")
    except Exception as e:
        print(f"[✗] 配置加载失败: {e}")
        return False
    
    try:
        from video_capture_thread import VideoCaptureThread, PerformanceMonitor
        print(f"[✓] 视频捕获模块加载成功")
    except Exception as e:
        print(f"[✗] 视频捕获模块加载失败: {e}")
        return False
    
    try:
        from optimized_gesture import OptimizedHandDetector, OptimizedGestureClassifier, AdaptiveGestureBuffer
        print(f"[✓] 手势识别模块加载成功")
    except Exception as e:
        print(f"[✗] 手势识别模块加载失败: {e}")
        return False
    
    try:
        from main_optimized import OptimizedGestureBattleGame
        print(f"[✓] 优化主程序加载成功")
    except Exception as e:
        print(f"[✗] 优化主程序加载失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("所有组件加载成功！")
    print("=" * 60)
    print("\n运行游戏命令:")
    print("  python main_optimized.py          # 性能优化版（推荐）")
    print("  python main.py                     # 原始版本")
    print("\n可选参数:")
    print("  --camera N        指定摄像头 ID (默认: 0)")
    print("  --debug           显示调试信息")
    print("  --detect-frames N 每N帧检测一次 (默认: 2)")
    print("  --no-multithread  禁用多线程（用于对比）")
    print("\n运行基准测试:")
    print("  python performance_test.py --benchmark")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='性能测试工具')
    parser.add_argument('--benchmark', action='store_true', help='运行完整基准测试')
    parser.add_argument('--camera', type=int, default=0, help='摄像头 ID (默认: 0)')
    parser.add_argument('--duration', type=float, default=3.0, help='每项测试持续时间 (默认: 3秒)')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark = PerformanceBenchmark()
        benchmark.run_all_benchmarks(args.camera, args.duration)
    else:
        quick_test()


if __name__ == '__main__':
    main()
