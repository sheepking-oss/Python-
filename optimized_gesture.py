import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from config import GameConfig, GestureConfig


class OptimizedHandDetector:
    def __init__(self, 
                 detect_every_n_frames=2,
                 inference_width=320,
                 inference_height=240,
                 min_detection_confidence=0.6,
                 min_tracking_confidence=0.5):
        
        self.detect_every_n_frames = detect_every_n_frames
        self.inference_width = inference_width
        self.inference_height = inference_height
        
        self.frame_count = 0
        self.last_detection_time = 0
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.cached_landmarks = None
        self.cached_image = None
        self.cached_results = None
        
        self.detection_times = deque(maxlen=30)
        self.detection_count = 0
        self.skip_count = 0
        
    def find_hands(self, image, draw=True, force_detect=False):
        h, w, _ = image.shape
        self.frame_count += 1
        
        should_detect = force_detect or (self.frame_count % self.detect_every_n_frames == 0)
        
        if should_detect:
            start_time = time.perf_counter()
            
            scale_x = self.inference_width / w
            scale_y = self.inference_height / h
            scale = min(scale_x, scale_y)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            small_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            image_rgb = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            self.cached_results = self.hands.process(image_rgb)
            self.cached_image = image.copy()
            
            detect_time = time.perf_counter() - start_time
            self.detection_times.append(detect_time)
            self.detection_count += 1
            
            if self.cached_results.multi_hand_landmarks:
                for hand_landmarks in self.cached_results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        landmark.x = landmark.x * (new_w / w)
                        landmark.y = landmark.y * (new_h / h)
        else:
            self.skip_count += 1
        
        if self.cached_results and self.cached_results.multi_hand_landmarks and draw:
            for hand_landmarks in self.cached_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        
        return image
    
    def find_position(self, image, hand_no=0):
        lm_list = []
        
        if self.cached_results and self.cached_results.multi_hand_landmarks:
            my_hand = self.cached_results.multi_hand_landmarks[hand_no]
            h, w, c = image.shape
            for id, lm in enumerate(my_hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
        
        return lm_list
    
    def get_stats(self):
        avg_time = sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0
        
        return {
            'detection_count': self.detection_count,
            'skip_count': self.skip_count,
            'avg_detection_time_ms': avg_time * 1000,
            'detect_every_n_frames': self.detect_every_n_frames,
            'inference_resolution': f'{self.inference_width}x{self.inference_height}'
        }


class OptimizedGestureClassifier:
    def __init__(self):
        self.finger_tip_ids = [4, 8, 12, 16, 20]
        self.last_gesture = None
        self.last_gesture_time = 0
        self.gesture_history = deque(maxlen=5)
        
        self.classification_count = 0
        self.classification_times = deque(maxlen=100)
    
    def classify(self, lm_list):
        if not lm_list:
            return None, 0.0
        
        start_time = time.perf_counter()
        
        fingers = []
        
        thumb_tip = lm_list[self.finger_tip_ids[0]]
        thumb_ip = lm_list[self.finger_tip_ids[0] - 1]
        thumb_mcp = lm_list[self.finger_tip_ids[0] - 2]
        
        palm_center_x = sum([lm_list[i][1] for i in [0, 5, 9, 13, 17]]) / 5
        
        if thumb_tip[1] < palm_center_x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for id in range(1, 5):
            tip_y = lm_list[self.finger_tip_ids[id]][2]
            pip_y = lm_list[self.finger_tip_ids[id] - 2][2]
            mcp_y = lm_list[self.finger_tip_ids[id] - 3][2]
            
            if tip_y < pip_y and tip_y < mcp_y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        total_fingers = fingers.count(1)
        
        gesture, confidence = self._finger_count_to_gesture(fingers, total_fingers, lm_list)
        
        self.gesture_history.append(gesture)
        
        if len(self.gesture_history) >= 3:
            gesture_counts = {}
            for g in self.gesture_history:
                if g:
                    gesture_counts[g] = gesture_counts.get(g, 0) + 1
            
            if gesture_counts:
                most_common = max(gesture_counts, key=gesture_counts.get)
                if gesture_counts[most_common] >= 2:
                    gesture = most_common
        
        self.classification_count += 1
        self.classification_times.append(time.perf_counter() - start_time)
        
        return gesture, confidence
    
    def _finger_count_to_gesture(self, fingers, total_fingers, lm_list):
        if total_fingers == 0:
            return 'fist', 0.95
        
        elif total_fingers == 5:
            return 'open', 0.95
        
        elif total_fingers == 2:
            if fingers[1] == 1 and fingers[2] == 1:
                if fingers[3] == 0 and fingers[4] == 0:
                    return 'scissors', 0.9
            return 'rock', 0.7
        
        elif total_fingers == 1:
            if fingers[1] == 1:
                return 'rock', 0.7
            return 'rock', 0.65
        
        elif total_fingers == 3:
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                return 'paper', 0.75
            return 'paper', 0.7
        
        elif total_fingers == 4:
            return 'paper', 0.85
        
        return 'rock', 0.6
    
    def get_stats(self):
        avg_time = sum(self.classification_times) / len(self.classification_times) if self.classification_times else 0
        
        return {
            'classification_count': self.classification_count,
            'avg_classification_time_ms': avg_time * 1000
        }


class AdaptiveGestureBuffer:
    def __init__(self, initial_buffer_size=5, min_buffer_size=3, max_buffer_size=10):
        self.initial_buffer_size = initial_buffer_size
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        self.buffer = []
        
        self.gesture_stability = {}
        self.last_stable_gesture = None
        self.stable_since = 0
        
        self.adaptive_threshold = 0.6
        self.stability_history = deque(maxlen=20)
    
    def add_gesture(self, gesture):
        if gesture:
            self.buffer.append(gesture)
            if len(self.buffer) > self.max_buffer_size:
                self.buffer.pop(0)
        
        return self.get_stable_gesture()
    
    def get_stable_gesture(self):
        if not self.buffer:
            return None
        
        gesture_counts = {}
        for g in self.buffer:
            if g:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        if not gesture_counts:
            return None
        
        most_common = max(gesture_counts, key=gesture_counts.get)
        count = gesture_counts[most_common]
        confidence = count / len(self.buffer)
        
        if confidence >= self.adaptive_threshold:
            if most_common != self.last_stable_gesture:
                self.last_stable_gesture = most_common
                self.stable_since = len(self.buffer)
            
            return most_common
        
        return None
    
    def update_stability(self, gesture_detected):
        if gesture_detected:
            self.stability_history.append(1)
        else:
            self.stability_history.append(0)
        
        if len(self.stability_history) >= 10:
            stability_ratio = sum(self.stability_history) / len(self.stability_history)
            
            if stability_ratio > 0.8:
                self.adaptive_threshold = 0.5
                self.max_buffer_size = 5
            elif stability_ratio > 0.5:
                self.adaptive_threshold = 0.6
                self.max_buffer_size = 7
            else:
                self.adaptive_threshold = 0.7
                self.max_buffer_size = 10
    
    def clear(self):
        self.buffer = []
        self.gesture_stability = {}
        self.last_stable_gesture = None
        self.stable_since = 0
        self.stability_history.clear()
    
    def get_stats(self):
        return {
            'buffer_size': len(self.buffer),
            'max_buffer_size': self.max_buffer_size,
            'adaptive_threshold': self.adaptive_threshold,
            'last_stable_gesture': self.last_stable_gesture
        }
