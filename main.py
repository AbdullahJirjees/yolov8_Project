import cv2
from ultralytics import YOLO

model = YOLO("../Yolo-Weights/yolov8l.pt")
results = model("images/2.jpg", show=True)
cv2.waitKey(0)