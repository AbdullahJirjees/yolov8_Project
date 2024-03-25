import cv2
from ultralytics import YOLO
import cvzone
import pandas as pd
import time

# Load model and class names
model = YOLO("./best.pt")  # Adjust path if needed
classNames = ['3', '4', '5', 'Helmet', 'boots', 'vest', 'head', 'helmet', 'no helmet', 'no vest', 'person', 'vest', 'vests']

# Open video source (adjust path or use webcam as needed)
cap = cv2.VideoCapture("people1.mp4")

# Create DataFrame for storing detections
df = pd.DataFrame(columns=['Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])

# Initialize variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()

    # Capture frame
    success, img = cap.read()

    if not success:
        print("Error reading frame from video stream.")
        break

    # Perform object detection
    results = model(img, stream=True)

    for r in results:
        for box in r.boxes:
            # Extract bounding box coordinates, confidence, and class
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            try:
                # Handle potential class index out of range
                if 0 <= cls < len(classNames):
                    # Draw bounding box and label using cvzone
                    cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1))
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                else:
                    print(f"Warning: Class index {cls} out of range. Using default label.")
                    cvzone.putTextRect(img, "Unknown Class", (max(0, x1), max(35, y1)), scale=1, thickness=1)

                # Add detection to DataFrame
                df = pd.concat([df, pd.DataFrame({'Class': classNames[cls], 'Confidence': conf, 'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2}, index=[0])], ignore_index=True)
            except IndexError:
                print("Error: Index out of range. Skipping detection.")

    # Calculate and display FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Image", img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Write detections to Excel file
df.to_excel('detections2.xlsx', index=False)

# Release resources
cap.release()
cv2.destroyAllWindows()
