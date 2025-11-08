import cv2
from ultralytics import YOLO
import numpy as np
import soundfile as sf
import os

model = YOLO('models/weights/best.pt')



class_names = model.names


cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


colors = np.random.uniform(0, 255, size=(len(class_names), 3))

print("Starting detection... Press 'q' to quit")

# Button coordinates and size
button_x, button_y, button_w, button_h = 50, 50, 150, 60
button_color = (0, 0, 255)
button_text = "describe"


generate_audio = 0

# Mouse callback function
def on_mouse(event, x, y, flags, param):
    global button_color, generate_audio
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is inside button area
        if button_x <= x <= button_x + button_w and button_y <= y <= button_y + button_h:
            generate_audio += 1
            button_color = (0, 255, 0)  # change color when pressed
    elif event == cv2.EVENT_LBUTTONUP:
        button_color = (0, 0, 255)  # return to red
cv2.namedWindow("YOLOv8 Detection")
cv2.setMouseCallback("YOLOv8 Detection", on_mouse)

while True:
    
   
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    # Draw button on the frame
    cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), button_color, -1)
    cv2.putText(frame, button_text, (button_x + 15, button_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    

    results = model(frame, conf=0.5)

    
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            

            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            
            class_name = class_names[cls]

            if generate_audio == 1:
                is_right = False
                is_left = False
                is_infront = False
                x_mean = (x1 + x2)/2
                y_mean = (y1 + y2)/2

                if x_mean < frame.shape[1] / 3:
                    is_left = True
                elif x_mean > frame.shape[1] * 2 / 3:
                    is_right = True
                if not is_left and not is_right:
                    is_infront = True

                print(f'There is a {class_name} {is_left* "on the left "}{is_right* "on the right"}{is_infront* "in front"}')
                
                audio_folder = "voice/"

                # Map phrases to audio files
                audio_map = {
                    "There-is-a": "There-is-a.wav",
                    "person": "Person.wav",
                    "On-the-right": "On-the-right.wav",
                    "On-the-left": "On-the-left.wav",
                    "In-front": "In-front.wav"
                }

                # Build the phrase dynamically
                audio_sequence = ["There-is-a", class_name]
                if is_left:
                    audio_sequence.append("On-the-left")
                if is_right:
                    audio_sequence.append("On-the-right")
                if is_infront:
                    audio_sequence.append("In-front")

                # Concatenate audio files
                final_audio = []
                sample_rate = None

                for phrase in audio_sequence:
                    file_path = os.path.join(audio_folder, audio_map[phrase])
                    audio_data, sr = sf.read(file_path)
                    
                    if sample_rate is None:
                        sample_rate = sr
                    elif sr != sample_rate:
                        raise ValueError(f"Sample rate mismatch: {sr} vs {sample_rate}")
                    
                    final_audio.append(audio_data)
                    
                    # Add 100ms silence (0.1 seconds)
                    silence = np.zeros(int(sr * 0.1))
                    final_audio.append(silence)

                # Concatenate all segments
                final_audio = np.concatenate(final_audio)

                # Export result
                output_path = os.path.join(os.getcwd(), "output_phrase.wav")
                sf.write(output_path, final_audio, sample_rate)
                os.system(f'aplay {output_path}') 
                generate_audio = 0


            
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
