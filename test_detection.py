from ultralytics import YOLO
import cv2

model = YOLO("model/yolov8n.pt")

img = cv2.imread("test.jpg")  # add a traffic image
results = model(img)

annotated = results[0].plot()
cv2.imshow("Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
