import cv2
from ultralytics import YOLO
import numpy as np
import os
import pyttsx3  # Import the pyttsx3 library

# Initialize the pyttsx3 engine
# It's recommended to initialize the engine outside the loop for better performance
engine = pyttsx3.init()
# Optional: Set properties like speech rate, volume, or voice
# engine.setProperty('rate', 150)  # Speed of speech
# engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

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
            button_color = (0, 255, 0)  # Change color when pressed
    elif event == cv2.EVENT_LBUTTONUP:
        button_color = (0, 0, 255)  # Return to red


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
    descriptions = []  # Store all detected descriptions for a single audio announcement

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            class_name = class_names[cls]

            # --- Start of pyttsx3 Integration ---
            if generate_audio > 0:
                x_mean = (x1 + x2) / 2

                # Determine object position relative to the frame
                if x_mean < frame.shape[1] / 3:
                    position = "on the left"
                elif x_mean > frame.shape[1] * 2 / 3:
                    position = "on the right"
                else:
                    position = "in front"

                # Construct the description phrase
                description_phrase = f'a {class_name} {position}'
                descriptions.append(description_phrase)
            # --- End of pyttsx3 Integration Prep ---

            # Drawing bounding box and label
            color = colors[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f'{class_name} {conf:.2f}'
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

    # --- pyttsx3 Execution ---
    if generate_audio > 0 and descriptions:
        # Create a single sentence from all detected objects
        if len(descriptions) == 1:
            full_sentence = f"There is {descriptions[0]}."
        elif len(descriptions) == 2:
            full_sentence = f"There is {descriptions[0]} and {descriptions[1]}."
        else:
            full_sentence = "There are " + ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}."

        print(f'Speaking: "{full_sentence}"')
        engine.say(full_sentence)
        engine.runAndWait()  # This blocks until the speech is complete
        generate_audio = 0

    elif generate_audio > 0 and not descriptions:
        engine.say("No objects detected.")
        engine.runAndWait()
        generate_audio = 0
    # --- End of pyttsx3 Execution ---

    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
