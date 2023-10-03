from ultralytics import YOLO
import cv2
import cvzone
import math


path = "static/test2.jpg"
img = cv2.imread(path)

model = YOLO("best.pt")

classNames = ['green', 'red', 'yellow']

def process_image(image):
    results = model(image, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # print(box)
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,255),3)
            cvzone.putTextRect(image, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1)
    cv2.imwrite('static/output.png', image)


# ----------- video ----------
def process_video(path, output_name = "output.mp4",output_path ="static/" ):
  cap = cv2.VideoCapture(path)  # For Video
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  if type(output_name)!=str :
    output_name = str(output_name)
  size = (frame_width, frame_height)
  ans = cv2.VideoWriter(output_path+output_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, size)
  while True:
    success, img = cap.read()
    if not success:
        break
    detections = model(img, stream=True)
    
    if not detections:
      continue
    for r in detections:
      boxes = r.boxes
      for box in boxes:
        cls = int(box.cls[0])
        conf = math.ceil((box.conf[0] * 100)) / 100
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cv2.rectangle(img, (x1-10, y1), (x2+10, y2), (255, 0, 255), 1)
        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1)
    ans.write(img)
  ans.release()
  cv2.destroyAllWindows()