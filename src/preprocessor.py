import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

class FacePreprocessor:
    """
    Handles face image loading, alignment/cropping via OpenCV SSD face detector,
    and preparation for network forward pass (resizing, tensor conversion, normalization, flip augmentation).
    """
    def __init__(self, detector_prototxt=None, detector_weights=None, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Initialize face detector if paths are provided and exist
        self.net = None
        if detector_prototxt and detector_weights:
            if os.path.exists(detector_prototxt) and os.path.exists(detector_weights):
                try:
                    self.net = cv2.dnn.readNetFromCaffe(detector_prototxt, detector_weights)
                except Exception as e:
                    print(f"Error loading Caffe face detector: {e}")
            else:
                print("Face detector weight or config file not found. Preprocessor will bypass face cropping.")

    def crop_face(self, img_pil):
        """
        Detects and crops the face with the highest confidence score.
        If no face is detected or detector is not initialized, returns the original image.
        """
        if self.net is None:
            return img_pil
            
        try:
            # Convert PIL image to OpenCV format (BGR)
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            h, w = img_cv.shape[:2]
            
            # Prepare image blob for SSD model input (expects 300x300)
            blob = cv2.dnn.blobFromImage(cv2.resize(img_cv, (300, 300)), 1.0, (300, 300), 
                                         (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()
            
            best_conf = 0.0
            best_box = None
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.confidence_threshold and confidence > best_conf:
                    best_conf = confidence
                    # Bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    best_box = box.astype("int")
            
            if best_box is not None:
                startX, startY, endX, endY = best_box
                # Add slight padding
                pad_w = int((endX - startX) * 0.1)
                pad_h = int((endY - startY) * 0.1)
                startX = max(0, startX - pad_w)
                startY = max(0, startY - pad_h)
                endX = min(w, endX + pad_w)
                endY = min(h, endY + pad_h)
                
                # Check for valid bounding box dimensions
                if endX - startX > 10 and endY - startY > 10:
                    cropped_cv = img_cv[startY:endY, startX:endX]
                    cropped_rgb = cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(cropped_rgb)
        except Exception as e:
            print(f"Bypassing face crop due to error: {e}")
            
        return img_pil

    def preprocess_image(self, img_pil, crop=True):
        """
        Crops (optional) and transforms the image to normalized PyTorch tensor format.
        
        Args:
            img_pil (PIL.Image): Input image.
            crop (bool): Whether to attempt face detection & cropping.
            
        Returns:
            torch.Tensor: Preprocessed image tensor shape (3, 112, 112).
        """
        if crop:
            img_pil = self.crop_face(img_pil)
        return self.transform(img_pil)

    def prepare_flip_pair(self, batch_tensor):
        """
        Generates standard and horizontally-flipped versions of the batch tensor.
        Used for evaluation flip-test augmentation.
        
        Args:
            batch_tensor (torch.Tensor): Tensor of shape (Batch, 3, 112, 112).
            
        Returns:
            tuple: (batch_tensor, flipped_batch_tensor)
        """
        flipped = torch.flip(batch_tensor, dims=[3])
        return batch_tensor, flipped
