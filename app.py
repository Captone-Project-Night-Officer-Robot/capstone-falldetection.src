import cv2
import cvzone
import math
from ultralytics import YOLO

cap = cv2.VideoCapture('F.mp4')

model = YOLO('yolov8s.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (980, 740))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (980, 740))

    results = model(frame, save=True)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            # implement fall detection using the coordinates x1,y1,x2
            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 10 and class_detect == 'person':
                cvzone.cornerRect(frame, [x1 + 2, y1 + 2, width + 2, height + 2], l=20, rt=1)
                cvzone.putTextRect(frame, f'{class_detect} {conf}%', [x1 + 8, y1 - 12], thickness=2, scale=1,
                                   colorR=None)

                if threshold < 0:
                    cvzone.putTextRect(frame, 'Fall Detected', [x1 + 10, y1 + 50], thickness=2, scale=4, border=3)

                else:
                    pass

    # Write frame to output video
    out.write(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
