import cv2
import mediapipe as mp
import numpy as np
from config import GameConfig, GestureConfig

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def find_hands(self, image, draw=True):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        self.results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True
        
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return image
    
    def find_position(self, image, hand_no=0):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            h, w, c = image.shape
            for id, lm in enumerate(my_hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
        return lm_list

class GestureClassifier:
    def __init__(self):
        self.finger_tip_ids = [4, 8, 12, 16, 20]
        
    def classify(self, lm_list):
        if not lm_list:
            return None, 0.0
        
        fingers = []
        
        if lm_list[self.finger_tip_ids[0]][1] < lm_list[self.finger_tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for id in range(1, 5):
            if lm_list[self.finger_tip_ids[id]][2] < lm_list[self.finger_tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        total_fingers = fingers.count(1)
        
        gesture, confidence = self._finger_count_to_gesture(fingers, total_fingers)
        
        return gesture, confidence
    
    def _finger_count_to_gesture(self, fingers, total_fingers):
        if total_fingers == 0:
            return 'fist', 0.95
        elif total_fingers == 5:
            return 'open', 0.95
        elif total_fingers == 2:
            if fingers[1] == 1 and fingers[2] == 1:
                return 'scissors', 0.9
            else:
                return 'rock', 0.7
        elif total_fingers == 1:
            return 'rock', 0.75
        elif total_fingers == 3:
            return 'paper', 0.8
        elif total_fingers == 4:
            return 'paper', 0.85
        else:
            return 'rock', 0.6

class GestureBuffer:
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.buffer = []
        
    def add_gesture(self, gesture):
        if gesture:
            self.buffer.append(gesture)
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)
        return self.get_most_common()
    
    def get_most_common(self):
        if not self.buffer:
            return None
        
        gesture_counts = {}
        for g in self.buffer:
            if g in gesture_counts:
                gesture_counts[g] += 1
            else:
                gesture_counts[g] = 1
        
        most_common = max(gesture_counts, key=gesture_counts.get)
        confidence = gesture_counts[most_common] / len(self.buffer)
        
        if confidence >= 0.6:
            return most_common
        return None
    
    def clear(self):
        self.buffer = []
