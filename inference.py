import torch
import cv2
import numpy as np
import argparse
import torchvision.transforms as transforms
from models import TransformerNet
from utils import get_device, tensor_to_image

class StyleTransferInference:
    def __init__(self, model_path, image_size=256, device=None):
        self.device = device if device is not None else get_device()
        self.image_size = image_size
        
        # Load Model Structure
        self.model = TransformerNet().to(self.device)
        
        # Load Weights
        print(f"Loading model from {model_path}...")
        try:
            # Check if it's a full checkpoint or just state_dict
            state_dict = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in state_dict:
                self.model.load_state_dict(state_dict['model_state_dict'])
            else:
                self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)
            
        self.model.eval()
        
        # Transform for Input (Same normalization as training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def process_frame(self, frame):
        original_h, original_w = frame.shape[:2]
        
        # 1. Preprocess: OpenCV (BGR) -> RGB -> PIL -> Tensor -> Normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        content_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        
        # 2. Inference
        with torch.no_grad():
            output_tensor = self.model(content_tensor)
        
        # 3. Postprocess: Tensor -> Numpy -> Denormalize -> RGB -> BGR
        output_image = tensor_to_image(output_tensor)
        
        # Resize back to original webcam size for display
        output_image = cv2.resize(output_image, (original_w, original_h))
        
        # RGB to BGR for OpenCV
        output_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        
        return output_bgr
    
    def process_webcam(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Could not open camera {camera_id}")
            return
            
        print("Webcam started! Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect (optional)
            frame = cv2.flip(frame, 1)
            
            stylized_frame = self.process_frame(frame)
            
            cv2.imshow('Style Transfer', stylized_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to .pth file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--camera-id', type=int, default=0)
    parser.add_argument('--image-size', type=int, default=480, help='Inference resolution')
    
    args = parser.parse_args()
    
    inference = StyleTransferInference(args.model, args.image_size)
    
    if args.webcam:
        inference.process_webcam(args.camera_id)

if __name__ == "__main__":
    main()