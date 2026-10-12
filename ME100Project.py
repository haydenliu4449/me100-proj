import cv2
from time import time
import requests
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, esp_ip="136.152.83.201", model_path="best.pt"):
        # --- ESP32 Configuration ---
        self.esp_ip = esp_ip
        self.signal_url = f"http://{self.esp_ip}/fire"
        self.last_signal_time = 0
        self.cooldown_seconds = 10
        
        # ---> Adjustable Confidence Threshold <---
        self.confidence_threshold = 0.30
        
        # --- Model Configuration ---
        print(f"Loading custom YOLOv11 model from {model_path}...")
        self.model = YOLO(model_path)

    def send_detection_signal(self):
        current_time = time()
        if (current_time - self.last_signal_time) > self.cooldown_seconds:
            print(f"Target acquired! Sending fire command to {self.signal_url}...")
            # Reset the cooldown timer
            self.last_signal_time = current_time
            try:
                # Send the signal to the ESP32
                response = requests.get(self.signal_url, timeout=1) 
                print(f"ESP32 Response: {response.text.strip()}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to reach ESP32. Error: {e}")

    def __call__(self):
        # Open the webcam
        player = cv2.VideoCapture(0)
        assert player.isOpened(), "Error reading video stream from webcam."
        
        print(f"Turret Vision Active. Minimum Confidence: {self.confidence_threshold*100}%. Press 'q' to quit.")
        
        while True:
            ret, frame = player.read()
            if not ret:
                break
                
            # Run YOLOv11 inference on the current frame
            results = self.model(frame, verbose=False)[0]
            
            # Loop through all the detections in the frame
            for box in results.boxes:
                conf = float(box.conf[0])
                
                # ---> CHECK 1: Is the confidence high enough? <---
                if conf < self.confidence_threshold:
                    continue
                    
                # ---> CHECK 2: Force the label and fire <---
                display_label = "Phone"
                bgr = (0, 0, 255) # Red target box
                
                # Trigger the firing sequence
                self.send_detection_signal()
                
                # Get the coordinates for the bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                # Draw the box and clean percentage label on the screen
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, f"{display_label} {conf*100:.0f}%", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
            
            # Show the live feed
            cv2.imshow('Nerf Turret Vision (YOLOv11)', frame)
            
            # Press 'q' to close the window safely
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        player.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = ObjectDetection(esp_ip="136.152.83.201", model_path="best.pt") 
    detector()