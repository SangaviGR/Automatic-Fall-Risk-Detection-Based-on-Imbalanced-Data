import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.previous_keypoints = {}
        
    def calculate_hw_ratio(self, keypoints: Dict) -> float:
        """Calculate Height-Width ratio as in equation (1) of the paper"""
        try:
            # Get y coordinates for height calculation
            y_coords = [keypoints[i][1] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8] if i in keypoints]
            # Get x coordinates for width calculation
            x_coords = [keypoints[i][0] for i in [2, 5, 8, 11, 14] if i in keypoints]
            
            if len(y_coords) < 2 or len(x_coords) < 2:
                return np.nan
                
            height = max(y_coords) - min(y_coords)
            width = max(x_coords) - min(x_coords)
            
            if width == 0:
                return np.nan
                
            return height / width
        except:
            return np.nan
    
    def calculate_spine_ratio(self, keypoints: Dict) -> float:
        """Calculate Spine ratio as in equation (2) of the paper"""
        try:
            # Keypoint 1 (neck) to keypoint 8 (mid hip) for spine length
            if 1 in keypoints and 8 in keypoints:
                spine_length = np.sqrt(
                    (keypoints[8][0] - keypoints[1][0])**2 + 
                    (keypoints[8][1] - keypoints[1][1])**2
                )
            else:
                return np.nan
                
            # Keypoint 9 (left hip) to keypoint 12 (right hip) for waist length
            if 9 in keypoints and 12 in keypoints:
                waist_length = np.sqrt(
                    (keypoints[12][0] - keypoints[9][0])**2 + 
                    (keypoints[12][1] - keypoints[9][1])**2
                )
            else:
                return np.nan
                
            if waist_length == 0:
                return np.nan
                
            return spine_length / waist_length
        except:
            return np.nan
    
    def calculate_distances(self, keypoints: Dict) -> Tuple[float, float]:
        """Calculate vertical distances as in equations (3) and (4)"""
        try:
            # Calculate feet position as midpoint of keypoints 11 and 14
            if 11 in keypoints and 14 in keypoints:
                feet_y = (keypoints[11][1] + keypoints[14][1]) / 2
            else:
                return np.nan, np.nan
                
            neck_to_feet = feet_y - keypoints[0][1] if 0 in keypoints else np.nan
            hip_to_feet = feet_y - keypoints[8][1] if 8 in keypoints else np.nan
            
            return neck_to_feet, hip_to_feet
        except:
            return np.nan, np.nan
    
    def calculate_acceleration(self, current_keypoints: Dict, person_id: int, frame_rate: int = 30) -> Tuple[float, float, float]:
        """Calculate acceleration features as in equations (5), (6), (7)"""
        try:
            if person_id not in self.previous_keypoints:
                self.previous_keypoints[person_id] = current_keypoints
                return np.nan, np.nan, np.nan
            
            prev_kp = self.previous_keypoints[person_id]
            time_interval = 1 / frame_rate
            
            accelerations = []
            for kp_id in [0, 1, 8]:  # Head, Neck, Hip
                if kp_id in current_keypoints and kp_id in prev_kp:
                    # Calculate vertical acceleration (negative y direction)
                    velocity_current = (current_keypoints[kp_id][1] - prev_kp[kp_id][1]) / time_interval
                    velocity_prev = (prev_kp[kp_id][1] - self.previous_keypoints.get(person_id, {}).get(kp_id, [0, 0])[1]) / time_interval
                    acceleration = (velocity_current - velocity_prev) / time_interval
                    accelerations.append(acceleration)
                else:
                    accelerations.append(np.nan)
            
            self.previous_keypoints[person_id] = current_keypoints
            return accelerations[0], accelerations[1], accelerations[2]
            
        except:
            return np.nan, np.nan, np.nan
    
    def calculate_deflection_angle(self, vector1: List, vector2: List) -> float:
        """Calculate deflection angle as in equation (8)"""
        try:
            vector1 = np.array(vector1)
            vector2 = np.array(vector2)
            
            dot_product = np.dot(vector1, vector2)
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return np.nan
                
            cosine_angle = dot_product / (norm1 * norm2)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            
            return np.degrees(np.arccos(cosine_angle))
        except:
            return np.nan
    
    def calculate_body_tilt_angle(self, keypoints: Dict) -> float:
        """Calculate body tilt angle as in equation (9)"""
        try:
            if 0 not in keypoints:  # Neck point
                return np.nan
                
            neck_x, neck_y = keypoints[0]
            body_parts = []
            
            # Mid hip
            if 8 in keypoints:
                body_parts.append(keypoints[8])
            # Mid knees (average of left and right knees)
            if 9 in keypoints and 10 in keypoints and 12 in keypoints and 13 in keypoints:
                mid_knee_x = (keypoints[10][0] + keypoints[13][0]) / 2
                mid_knee_y = (keypoints[10][1] + keypoints[13][1]) / 2
                body_parts.append([mid_knee_x, mid_knee_y])
            # Mid ankles
            if 11 in keypoints and 14 in keypoints:
                mid_ankle_x = (keypoints[11][0] + keypoints[14][0]) / 2
                mid_ankle_y = (keypoints[11][1] + keypoints[14][1]) / 2
                body_parts.append([mid_ankle_x, mid_ankle_y])
            
            if not body_parts:
                return np.nan
                
            # Calculate angles and return the smallest one as per paper
            angles = []
            for part in body_parts:
                delta_x = part[0] - neck_x
                delta_y = part[1] - neck_y
                
                if delta_x == 0:
                    angle = 90.0
                else:
                    angle = np.degrees(np.arctan2(delta_y, delta_x))
                
                angles.append(abs(angle))
            
            return min(angles) if angles else np.nan
            
        except:
            return np.nan
    
    def extract_all_features(self, keypoints: Dict, person_id: int) -> Dict:
        """Extract all features from keypoints for one person"""
        features = {}
        
        # Ratio features
        features['hw_ratio'] = self.calculate_hw_ratio(keypoints)
        features['spine_ratio'] = self.calculate_spine_ratio(keypoints)
        
        # Distance features
        neck_dist, hip_dist = self.calculate_distances(keypoints)
        features['neck_to_feet_dist'] = neck_dist
        features['hip_to_feet_dist'] = hip_dist
        
        # Acceleration features
        head_acc, neck_acc, hip_acc = self.calculate_acceleration(keypoints, person_id)
        features['head_acceleration'] = head_acc
        features['neck_acceleration'] = neck_acc
        features['hip_acceleration'] = hip_acc
        
        # Deflection and tilt features
        features['body_tilt_angle'] = self.calculate_body_tilt_angle(keypoints)
        
        return features