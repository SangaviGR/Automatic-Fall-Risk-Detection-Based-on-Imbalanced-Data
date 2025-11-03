import cv2
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from utils.feature_extractor import FeatureExtractor

class PoseDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.feature_extractor = FeatureExtractor()
        
        # Initialize OpenPose (you'll need to install OpenPose separately)
        self.setup_openpose()
    
    def setup_openpose(self):
        """Setup OpenPose detector"""
        try:
            # Option 1: Using OpenPose Python wrapper
            try:
                from openpose import pyopenpose as op
                op_params = {
                    "model_folder": self.config['openpose']['model_folder'],
                    "net_resolution": self.config['openpose']['net_resolution'],
                    "model_pose": self.config['openpose']['model_pose']
                }
                self.op_wrapper = op.WrapperPython()
                self.op_wrapper.configure(op_params)
                self.op_wrapper.start()
                self.use_openpose = True
            except ImportError:
                # Option 2: Using OpenCV's DNN module with pre-trained models
                self.use_openpose = False
                print("OpenPose not available, using alternative pose estimation")
                
        except Exception as e:
            print(f"Error setting up OpenPose: {e}")
            self.use_openpose = False
    
    def detect_poses(self, image: np.ndarray) -> List[Dict[int, Tuple[float, float]]]:
        """Detect poses in image and return keypoints"""
        if self.use_openpose:
            return self._detect_with_openpose(image)
        else:
            return self._detect_with_opencv(image)
    
    def _detect_with_openpose(self, image: np.ndarray) -> List[Dict[int, Tuple[float, float]]]:
        """Detect poses using OpenPose"""
        try:
            from openpose import pyopenpose as op
            datum = op.Datum()
            datum.cvInputData = image
            self.op_wrapper.emplaceAndPop([datum])
            
            keypoints_list = []
            if datum.poseKeypoints is not None:
                for person in datum.poseKeypoints:
                    keypoints = {}
                    for i, point in enumerate(person):
                        if point[2] > 0.1:  # Confidence threshold
                            keypoints[i] = (float(point[0]), float(point[1]))
                    keypoints_list.append(keypoints)
            
            return keypoints_list
        except Exception as e:
            print(f"Error in OpenPose detection: {e}")
            return []
    
    def _detect_with_opencv(self, image: np.ndarray) -> List[Dict[int, Tuple[float, float]]]:
        """Alternative pose detection using OpenCV DNN"""
        # This is a simplified version - you would need to implement
        # or use a pre-trained pose estimation model
        print("Alternative pose detection not implemented")
        return []
    
    def extract_features_from_image(self, image: np.ndarray) -> List[Dict[str, float]]:
        """Extract features from all detected persons in image"""
        keypoints_list = self.detect_poses(image)
        features_list = []
        
        for keypoints in keypoints_list:
            features = self.feature_extractor.extract_features(keypoints)
            features_list.append(features)
        
        return features_list
    
    def process_video(self, video_path: str, output_path: str = None):
        """Process video and extract features frame by frame"""
        cap = cv2.VideoCapture(video_path)
        all_features = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            features = self.extract_features_from_image(frame)
            all_features.extend(features)
            
            if output_path:
                # You can add visualization here
                pass
        
        cap.release()
        return all_features