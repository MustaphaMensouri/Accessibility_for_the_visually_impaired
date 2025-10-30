import cv2
from ultralytics import YOLO
import numpy as np


model = YOLO('models/weights/best.pt')


class_names = model.names


cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


colors = np.random.uniform(0, 255, size=(len(class_names), 3))

print("Starting detection... Press 'q' to quit")

while True:
   
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    
    results = model(frame, conf=0.5)
    
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            
            class_name = class_names[cls]
            
            
            color = colors[cls]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            
            label = f'{class_name} {conf:.2f}'
            
            
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            
            cv2.rectangle(
                frame, 
                (x1, y1 - label_h - 10), 
                (x1 + label_w, y1), 
                color, 
                -1
            )
            
           
            cv2.putText(
                frame, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
    
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(
        frame, 
        f'FPS: {fps:.1f}', 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )
    
    
    cv2.imshow('YOLOv8 Detection', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
