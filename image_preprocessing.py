import cv2
import numpy as np
from collections import deque
import time


class ImagePreprocessor:
    def __init__(self, 
                 blur_kernel=5,
                 clahe_clip_limit=2.0,
                 clahe_grid_size=8,
                 enable_preprocessing=True):
        
        self.blur_kernel = blur_kernel
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.enable_preprocessing = enable_preprocessing
        
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_grid_size, clahe_grid_size)
        )
        
        self.preprocess_times = deque(maxlen=100)
        self.preprocess_count = 0
        
        self.original_frame = None
        self.processed_frame = None
        self.debug_info = {}
    
    def preprocess(self, image, roi=None):
        if not self.enable_preprocessing:
            return image
        
        start_time = time.perf_counter()
        
        self.original_frame = image.copy()
        self.debug_info = {}
        
        if roi is not None:
            x, y, w, h = roi
            roi_image = image[y:y+h, x:x+w].copy()
        else:
            roi_image = image.copy()
        
        processed = self._apply_preprocessing_pipeline(roi_image)
        
        if roi is not None:
            result = image.copy()
            result[y:y+h, x:x+w] = processed
            self.processed_frame = result
        else:
            self.processed_frame = processed
            result = processed
        
        self.preprocess_times.append(time.perf_counter() - start_time)
        self.preprocess_count += 1
        
        return result
    
    def _apply_preprocessing_pipeline(self, image):
        self.debug_info['original_shape'] = image.shape
        
        denoised = self._apply_gaussian_blur(image)
        self.debug_info['denoised'] = denoised
        
        enhanced = self._apply_illumination_correction(denoised)
        self.debug_info['enhanced'] = enhanced
        
        contrast_enhanced = self._apply_contrast_enhancement(enhanced)
        self.debug_info['contrast_enhanced'] = contrast_enhanced
        
        return contrast_enhanced
    
    def _apply_gaussian_blur(self, image):
        if self.blur_kernel > 1:
            kernel_size = (self.blur_kernel, self.blur_kernel)
            blurred = cv2.GaussianBlur(image, kernel_size, 0)
            return blurred
        return image
    
    def _apply_illumination_correction(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        l_clahe = self.clahe.apply(l)
        
        lab_clahe = cv2.merge((l_clahe, a, b))
        corrected = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return corrected
    
    def _apply_contrast_enhancement(self, image):
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        y_eq = cv2.equalizeHist(y)
        
        ycrcb_eq = cv2.merge((y_eq, cr, cb))
        enhanced = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
        
        return enhanced
    
    def extract_hand_roi(self, image, landmarks, padding=50):
        if not landmarks:
            return None, None
        
        h, w = image.shape[:2]
        
        x_coords = [lm[1] for lm in landmarks]
        y_coords = [lm[2] for lm in landmarks]
        
        min_x = max(0, min(x_coords) - padding)
        max_x = min(w, max(x_coords) + padding)
        min_y = max(0, min(y_coords) - padding)
        max_y = min(h, max(y_coords) + padding)
        
        roi_width = max_x - min_x
        roi_height = max_y - min_y
        
        if roi_width <= 0 or roi_height <= 0:
            return None, None
        
        roi = (min_x, min_y, roi_width, roi_height)
        roi_image = image[min_y:max_y, min_x:max_x].copy()
        
        return roi, roi_image
    
    def get_stats(self):
        if self.preprocess_times:
            avg_time = sum(self.preprocess_times) / len(self.preprocess_times)
            return {
                'preprocess_count': self.preprocess_count,
                'avg_preprocess_time_ms': avg_time * 1000,
                'blur_kernel': self.blur_kernel,
                'clahe_clip_limit': self.clahe_clip_limit
            }
        return {
            'preprocess_count': 0,
            'avg_preprocess_time_ms': 0
        }


class SkinDetector:
    def __init__(self, 
                 ycrcb_lower=(0, 133, 77),
                 ycrcb_upper=(255, 173, 127),
                 hsv_lower=(0, 10, 60),
                 hsv_upper=(20, 150, 255),
                 use_adaptive_threshold=True):
        
        self.ycrcb_lower = np.array(ycrcb_lower, dtype=np.uint8)
        self.ycrcb_upper = np.array(ycrcb_upper, dtype=np.uint8)
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        
        self.use_adaptive_threshold = use_adaptive_threshold
        
        self.kernel_3 = np.ones((3, 3), np.uint8)
        self.kernel_5 = np.ones((5, 5), np.uint8)
        self.kernel_7 = np.ones((7, 7), np.uint8)
        
        self.detection_times = deque(maxlen=100)
        self.detection_count = 0
        
        self.avg_skin_color = None
        self.skin_color_history = deque(maxlen=30)
    
    def detect(self, image, roi=None):
        start_time = time.perf_counter()
        
        if roi is not None:
            x, y, w, h = roi
            roi_image = image[y:y+h, x:x+w].copy()
        else:
            roi_image = image.copy()
        
        ycrcb_mask = self._detect_skin_ycrcb(roi_image)
        hsv_mask = self._detect_skin_hsv(roi_image)
        
        combined_mask = cv2.bitwise_and(ycrcb_mask, hsv_mask)
        
        refined_mask = self._refine_mask(combined_mask)
        
        if self.use_adaptive_threshold:
            adaptive_mask = self._adaptive_skin_detection(roi_image)
            refined_mask = cv2.bitwise_or(refined_mask, adaptive_mask)
        
        skin_pixels = cv2.countNonZero(refined_mask)
        total_pixels = refined_mask.shape[0] * refined_mask.shape[1]
        skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0
        
        self.detection_times.append(time.perf_counter() - start_time)
        self.detection_count += 1
        
        result = {
            'mask': refined_mask,
            'skin_ratio': skin_ratio,
            'skin_pixels': skin_pixels,
            'total_pixels': total_pixels
        }
        
        return result
    
    def _detect_skin_ycrcb(self, image):
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)
        return mask
    
    def _detect_skin_hsv(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        return mask
    
    def _refine_mask(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_3)
        mask = cv2.dilate(mask, self.kernel_5, iterations=1)
        mask = cv2.erode(mask, self.kernel_3, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            refined_mask = np.zeros_like(mask)
            cv2.drawContours(refined_mask, [largest_contour], 0, 255, -1)
            
            hull = cv2.convexHull(largest_contour)
            cv2.drawContours(refined_mask, [hull], 0, 255, -1)
            
            return refined_mask
        
        return mask
    
    def _adaptive_skin_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return adaptive_thresh
    
    def apply_background_mask(self, image, mask):
        result = image.copy()
        background = np.zeros_like(image)
        result[mask == 0] = background[mask == 0]
        return result
    
    def isolate_hand(self, image, mask):
        isolated = cv2.bitwise_and(image, image, mask=mask)
        return isolated
    
    def get_hand_contour_features(self, mask):
        features = {}
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return features
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(largest_contour)
        features['area'] = area
        
        perimeter = cv2.arcLength(largest_contour, True)
        features['perimeter'] = perimeter
        
        if area > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            features['circularity'] = circularity
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        features['bounding_rect'] = (x, y, w, h)
        features['aspect_ratio'] = w / h if h > 0 else 0
        
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        features['convex_hull_area'] = hull_area
        
        if hull_area > 0:
            solidity = area / hull_area
            features['solidity'] = solidity
        
        try:
            defects = cv2.convexityDefects(
                largest_contour, cv2.convexHull(largest_contour, returnPoints=False)
            )
            if defects is not None:
                finger_count = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    if d > 10000:
                        finger_count += 1
                features['finger_count'] = finger_count
                features['defect_count'] = defects.shape[0]
        except:
            pass
        
        return features
    
    def get_stats(self):
        if self.detection_times:
            avg_time = sum(self.detection_times) / len(self.detection_times)
            return {
                'detection_count': self.detection_count,
                'avg_detection_time_ms': avg_time * 1000
            }
        return {
            'detection_count': 0,
            'avg_detection_time_ms': 0
        }


class AdvancedGestureClassifier:
    def __init__(self, 
                 use_skin_detection=True,
                 use_contour_features=True,
                 min_skin_ratio=0.1):
        
        self.use_skin_detection = use_skin_detection
        self.use_contour_features = use_contour_features
        self.min_skin_ratio = min_skin_ratio
        
        self.finger_tip_ids = [4, 8, 12, 16, 20]
        
        self.gesture_history = deque(maxlen=5)
        self.classification_count = 0
        
        self.skin_detector = SkinDetector()
        self.last_skin_ratio = 0.0
    
    def classify(self, landmarks, image=None, roi=None):
        if not landmarks:
            return None, 0.0
        
        if self.use_skin_detection and image is not None:
            skin_result = self.skin_detector.detect(image, roi)
            self.last_skin_ratio = skin_result['skin_ratio']
            
            if skin_result['skin_ratio'] < self.min_skin_ratio:
                return None, 0.0
        
        fingers = self._count_fingers(landmarks)
        total_fingers = fingers.count(1)
        
        contour_features = {}
        if self.use_contour_features and image is not None and roi is not None:
            x, y, w, h = roi
            roi_image = image[y:y+h, x:x+w].copy()
            skin_result = self.skin_detector.detect(image, roi)
            contour_features = self.skin_detector.get_hand_contour_features(skin_result['mask'])
        
        gesture, confidence = self._classify_from_fingers(
            fingers, total_fingers, contour_features, landmarks
        )
        
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
                    confidence = min(1.0, confidence + 0.1)
        
        self.classification_count += 1
        
        return gesture, confidence
    
    def _count_fingers(self, landmarks):
        fingers = []
        
        palm_center_x = sum([landmarks[i][1] for i in [0, 5, 9, 13, 17]]) / 5
        palm_center_y = sum([landmarks[i][2] for i in [0, 5, 9, 13, 17]]) / 5
        
        thumb_tip = landmarks[self.finger_tip_ids[0]]
        thumb_ip = landmarks[self.finger_tip_ids[0] - 1]
        thumb_mcp = landmarks[self.finger_tip_ids[0] - 2]
        
        if thumb_tip[1] < palm_center_x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for id in range(1, 5):
            tip_y = landmarks[self.finger_tip_ids[id]][2]
            pip_y = landmarks[self.finger_tip_ids[id] - 2][2]
            mcp_y = landmarks[self.finger_tip_ids[id] - 3][2]
            
            if tip_y < pip_y and tip_y < mcp_y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def _classify_from_fingers(self, fingers, total_fingers, contour_features, landmarks):
        if total_fingers == 0:
            return 'fist', 0.95
        
        elif total_fingers == 5:
            return 'open', 0.95
        
        elif total_fingers == 2:
            if fingers[1] == 1 and fingers[2] == 1:
                if fingers[3] == 0 and fingers[4] == 0:
                    angle = self._calculate_angle(landmarks, 0, 5, 8)
                    if 30 < angle < 120:
                        return 'scissors', 0.92
                    return 'scissors', 0.85
            return 'rock', 0.7
        
        elif total_fingers == 1:
            if fingers[1] == 1:
                return 'rock', 0.75
            return 'rock', 0.65
        
        elif total_fingers == 3:
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                return 'paper', 0.75
            return 'paper', 0.7
        
        elif total_fingers == 4:
            return 'paper', 0.85
        
        return 'rock', 0.6
    
    def _calculate_angle(self, landmarks, p1, p2, p3):
        point1 = np.array([landmarks[p1][1], landmarks[p1][2]])
        point2 = np.array([landmarks[p2][1], landmarks[p2][2]])
        point3 = np.array([landmarks[p3][1], landmarks[p3][2]])
        
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def validate_hand_presence(self, image, landmarks, roi=None):
        if not landmarks:
            return False, 0.0
        
        if self.use_skin_detection:
            skin_result = self.skin_detector.detect(image, roi)
            
            if skin_result['skin_ratio'] < self.min_skin_ratio:
                return False, skin_result['skin_ratio']
            
            return True, skin_result['skin_ratio']
        
        return True, 1.0
    
    def get_stats(self):
        return {
            'classification_count': self.classification_count,
            'use_skin_detection': self.use_skin_detection,
            'use_contour_features': self.use_contour_features,
            'last_skin_ratio': self.last_skin_ratio
        }
