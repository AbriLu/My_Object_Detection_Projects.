import cv2
from ultralytics import YOLO as yolo
import cv2 as cv
import cvzone
import math
from sort import *

cap = cv.VideoCapture("../videos/people.mp4") # Selecting the video input method in this case a video from our hard drive

model = yolo("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv.imread("detection_area (Custom).png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)  # Setting up the tracker
limits = []
limitsDown = [447, 250, 635, 250]
limitsUp = [195, 350, 440, 350]
numOfObjectsUp = 0
numOfObjectsDown = 0
numOfObjectsTotal = 0
listOfUniqueIdsDown = []
listOfUniqueIdsUp = []

while True:
    success, img = cap.read()
    imgRegion = cv.bitwise_and(img, mask)  # Combining the mask and the video
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))  # Array of detections with attributes x1, y1, x2, y2 and the tracking ID

    for r in results:  # looping through images
        boxes = r.boxes
        for box in boxes:  # Looping through bounding boxes in the image 'r'

            # Using the CVZONE package to draw bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]  # The index '0' is meant to extract the data from the box.xyxy tensor object
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox = x1, y1, x2, y2
            print(x1, y1, x2-x1, y2-y1)

            # Displaying the object class and confidence value using CVZONE
            conf = math.ceil(box.conf*100)/100
            print(f"Confidence = {conf}")
            class_index = box.cls
            objectClass = classNames[int(class_index[0])]
            print(f"Class Name = {objectClass}")

            # Selecting which object to detect and printing the bounding boxes on them
            if objectClass == 'person' and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Drawing (lines for) the counting threshold area

    cv.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)
    cv.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)

    # Drawing bounding boxes and setting labels on detected objects

    resultTracker = tracker.update(detections)
    for result in resultTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2, Id = int(x1), int(y1), int(x2), int(y2), int(Id)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2, colorR=(255, 0, 255))
        print(result)
        cvzone.putTextRect(img,
                           f'{Id}',
                           (max(0, x1), max(35, y1)),
                           scale=1, thickness=1,
                           offset=3)  # This line prints the detection Id on the bounding boxes
        cx, cy = int(x1 + w / 2), int(y1 + h / 2)
        cv.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Setting the counting area in the image and counting the number of objects going down

        if limitsDown[0] < cx < limitsDown[2] and (limitsDown[1]-20) < cy < (limitsDown[1]+20):  # For all the objects in this area, if the
            # center of an object is in this area of the image then count the object
            if Id not in listOfUniqueIdsDown:
                listOfUniqueIdsDown.append(Id)
                cv.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
            numOfObjectsDown = len(listOfUniqueIdsDown)

            # Setting the counting area in the image and counting the number of objects going up

        if limitsUp[0] < cx < limitsUp[2] and (limitsUp[1] - 20) < cy < (limitsUp[1] + 20):  # For all the objects in this area, if the
            # center of an object is in this area of the image then count the object
            if Id not in listOfUniqueIdsUp:
                listOfUniqueIdsUp.append(Id)
                cv.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)
            numOfObjectsUp = len(listOfUniqueIdsUp)

    numOfObjectsTotal = numOfObjectsDown + numOfObjectsUp
    cvzone.putTextRect(img, f'Count={numOfObjectsTotal}', (50, 50))  # Printing the number of objects detected

    cv.imshow("Image", img)
    # cv.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)

