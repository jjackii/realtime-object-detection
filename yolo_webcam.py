import sys
import numpy as np
import cv2

url = "http://192.168.219.116:8080/video"
cap = cv2.VideoCapture(url)

# 모델 & 설정 파일
model = 'final_project\yolo_v3\yolov3.weights'
config = 'final_project\yolo_v3\yolov3.cfg'
class_labels = 'final_project\yolo_v3\coco.names'
confThreshold = 0.5
nmsThreshold = 0.4

# # 테스트 이미지 파일
# frame_files = ['yolo_v3/dog.jpg', 'yolo_v3/person.jpg', 
#              'yolo_v3/sheep.jpg', 'yolo_v3/kite.jpg']

# 네트워크 생성
net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()

# 클래스 이름 불러오기

classes = []
with open(class_labels, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 출력 레이어 이름 받아오기

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# output_layers = ['yolo_82', 'yolo_94', 'yolo_106']

# 실행

while True:

    ret, frame = cap.read()

    if not ret:
        break
    # 블롭 생성 & 추론
    blob = cv2.dnn.blobFromImage(frame, 1/255., (320, 320), swapRB=True)
    net.setInput(blob)
    outs = net.forward(output_layers) #

    h, w = frame.shape[:2]

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            # detection: 4(bounding box) + 1(objectness_score) + 80(class confidence)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThreshold:
                # 바운딩 박스 중심 좌표 & 박스 크기
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)

                # 바운딩 박스 좌상단 좌표
                sx = int(cx - bw / 2)
                sy = int(cy - bh / 2)

                boxes.append([sx, sy, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    # 비최대 억제, Non Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        sx, sy, bw, bh = boxes[i]
        label = f'{classes[class_ids[i]]}: {confidences[i]:.2}'
        color = colors[class_ids[i]]
        cv2.rectangle(frame, (sx, sy, bw, bh), color, 2)
        cv2.putText(frame, label, (sx, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 1, cv2.LINE_AA)

    

    
    dst = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)

    cv2.imshow('frame', dst)
    
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
