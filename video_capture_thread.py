import cv2
import threading
import time
import queue
from collections import deque
from config import GameConfig


class VideoCaptureThread:
    def __init__(self, camera_id=0, width=1280, height=720, fps=30, buffer_size=1):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        self.buffer_size = buffer_size
        
        self.cap = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        self.latest_frame = None
        self.frame_count = 0
        self.last_frame_time = 0
        self.actual_fps = 0
        
        self.read_times = deque(maxlen=30)
        self.frame_drops = 0
        
    def init_camera(self):
        print(f"[VideoCaptureThread] 正在初始化摄像头 (ID: {self.camera_id})...")
        
        backend = cv2.CAP_DSHOW if cv2.__version__.startswith('4') else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.camera_id, backend)
        
        if not self.cap.isOpened():
            print(f"[VideoCaptureThread] 尝试使用默认后端...")
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise Exception(f"无法打开摄像头 ID: {self.camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[VideoCaptureThread] 摄像头初始化成功！")
        print(f"[VideoCaptureThread]   分辨率: {actual_width}x{actual_height}")
        print(f"[VideoCaptureThread]   目标帧率: {self.target_fps} (实际: {actual_fps})")
        print(f"[VideoCaptureThread]   缓冲区大小: {self.buffer_size}")
        
        return True
    
    def start(self):
        if self.running:
            return True
        
        if not self.cap:
            self.init_camera()
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("[VideoCaptureThread] 视频捕获线程已启动")
        
        return True
    
    def stop(self):
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        print("[VideoCaptureThread] 视频捕获线程已停止")
        print(f"[VideoCaptureThread]   总帧数: {self.frame_count}")
        print(f"[VideoCaptureThread]   丢帧数: {self.frame_drops}")
    
    def _capture_loop(self):
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            start_time = time.perf_counter()
            
            if self.cap.grab():
                ret, frame = self.cap.retrieve()
                
                if ret and frame is not None:
                    consecutive_failures = 0
                    
                    with self.lock:
                        if self.latest_frame is not None:
                            self.frame_drops += 1
                        self.latest_frame = frame.copy()
                        self.frame_count += 1
                    
                    self.read_times.append(time.perf_counter() - start_time)
                    
                    if self.last_frame_time > 0:
                        interval = start_time - self.last_frame_time
                        if interval > 0:
                            current_fps = 1.0 / interval
                            self.actual_fps = 0.9 * self.actual_fps + 0.1 * current_fps
                    
                    self.last_frame_time = start_time
                else:
                    consecutive_failures += 1
            else:
                consecutive_failures += 1
                time.sleep(0.001)
            
            if consecutive_failures >= max_failures:
                print(f"[VideoCaptureThread] 警告: 连续 {max_failures} 次读取失败")
                consecutive_failures = 0
    
    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()
    
    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()
    
    def get_stats(self):
        with self.lock:
            avg_read_time = sum(self.read_times) / len(self.read_times) if self.read_times else 0
            
            return {
                'frame_count': self.frame_count,
                'frame_drops': self.frame_drops,
                'actual_fps': self.actual_fps,
                'avg_read_time_ms': avg_read_time * 1000,
                'buffer_used': 1 if self.latest_frame is not None else 0
            }


class FrameProcessor:
    def __init__(self, process_func=None, max_workers=1, queue_size=2):
        self.process_func = process_func
        self.max_workers = max_workers
        self.queue_size = queue_size
        
        self.input_queue = queue.Queue(maxsize=queue_size)
        self.output_queue = queue.Queue(maxsize=queue_size)
        
        self.running = False
        self.workers = []
        self.lock = threading.Lock()
        
        self.process_count = 0
        self.process_times = deque(maxlen=30)
        
    def start(self):
        if self.running:
            return
        
        self.running = True
        
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        
        print(f"[FrameProcessor] 已启动 {self.max_workers} 个处理线程")
    
    def stop(self):
        self.running = False
        
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1.0)
        
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
        
        print(f"[FrameProcessor] 处理线程已停止，总处理数: {self.process_count}")
    
    def _worker_loop(self, worker_id):
        while self.running:
            try:
                frame, frame_id, extra_data = self.input_queue.get(timeout=0.1)
                
                start_time = time.perf_counter()
                
                if self.process_func:
                    result = self.process_func(frame, extra_data)
                else:
                    result = frame
                
                process_time = time.perf_counter() - start_time
                
                with self.lock:
                    self.process_times.append(process_time)
                    self.process_count += 1
                
                try:
                    self.output_queue.put_nowait((result, frame_id, extra_data))
                except queue.Full:
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait((result, frame_id, extra_data))
                    except queue.Empty:
                        pass
                
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[FrameProcessor] Worker {worker_id} 错误: {e}")
    
    def submit(self, frame, frame_id=None, extra_data=None):
        if frame_id is None:
            frame_id = time.perf_counter()
        
        try:
            if self.input_queue.full():
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.input_queue.put_nowait((frame, frame_id, extra_data))
            return True
        except queue.Full:
            return False
    
    def get_result(self, timeout=0.001):
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self):
        with self.lock:
            avg_time = sum(self.process_times) / len(self.process_times) if self.process_times else 0
            
            return {
                'process_count': self.process_count,
                'avg_process_time_ms': avg_time * 1000,
                'input_queue_size': self.input_queue.qsize(),
                'output_queue_size': self.output_queue.qsize()
            }


class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = {
            'capture_time': deque(maxlen=window_size),
            'process_time': deque(maxlen=window_size),
            'render_time': deque(maxlen=window_size),
            'total_time': deque(maxlen=window_size),
            'fps': deque(maxlen=window_size)
        }
        self.last_frame_time = time.perf_counter()
    
    def start_frame(self):
        self.frame_start = time.perf_counter()
        
        if self.last_frame_time > 0:
            interval = self.frame_start - self.last_frame_time
            if interval > 0:
                self.metrics['fps'].append(1.0 / interval)
        
        self.last_frame_time = self.frame_start
    
    def record_capture(self):
        self.capture_end = time.perf_counter()
        self.metrics['capture_time'].append(self.capture_end - self.frame_start)
    
    def record_process(self):
        self.process_end = time.perf_counter()
        self.metrics['process_time'].append(self.process_end - self.capture_end)
    
    def record_render(self):
        self.render_end = time.perf_counter()
        self.metrics['render_time'].append(self.render_end - self.process_end)
        self.metrics['total_time'].append(self.render_end - self.frame_start)
    
    def get_stats(self):
        stats = {}
        for key, values in self.metrics.items():
            if values:
                stats[f'{key}_avg_ms'] = (sum(values) / len(values)) * 1000
                stats[f'{key}_min_ms'] = min(values) * 1000
                stats[f'{key}_max_ms'] = max(values) * 1000
                stats[f'{key}_count'] = len(values)
            else:
                stats[f'{key}_avg_ms'] = 0
                stats[f'{key}_min_ms'] = 0
                stats[f'{key}_max_ms'] = 0
                stats[f'{key}_count'] = 0
        return stats
    
    def format_stats(self):
        stats = self.get_stats()
        lines = [
            "=" * 60,
            "性能统计",
            "=" * 60,
            f"FPS:      平均: {stats.get('fps_avg_ms', 0) * 1000:.1f}  "
            f"范围: {stats.get('fps_min_ms', 0) * 1000:.1f} - {stats.get('fps_max_ms', 0) * 1000:.1f}",
            f"捕获耗时: 平均: {stats.get('capture_time_avg_ms', 0):.2f}ms  "
            f"范围: {stats.get('capture_time_min_ms', 0):.2f} - {stats.get('capture_time_max_ms', 0):.2f}",
            f"处理耗时: 平均: {stats.get('process_time_avg_ms', 0):.2f}ms  "
            f"范围: {stats.get('process_time_min_ms', 0):.2f} - {stats.get('process_time_max_ms', 0):.2f}",
            f"渲染耗时: 平均: {stats.get('render_time_avg_ms', 0):.2f}ms  "
            f"范围: {stats.get('render_time_min_ms', 0):.2f} - {stats.get('render_time_max_ms', 0):.2f}",
            f"总耗时:   平均: {stats.get('total_time_avg_ms', 0):.2f}ms  "
            f"范围: {stats.get('total_time_min_ms', 0):.2f} - {stats.get('total_time_max_ms', 0):.2f}",
            "=" * 60
        ]
        return "\n".join(lines)
