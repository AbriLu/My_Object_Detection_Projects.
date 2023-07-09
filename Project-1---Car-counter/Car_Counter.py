import cv2
import numpy
from ultralytics import YOLO as yolo
import cv2 as cv
import cvzone
import math
from sort import *

cap = cv.VideoCapture("../videos/cars.mp4") # Selecting the video input method in this case a video from our hard drive

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


# Object tracking setup
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [320, 315, 673, 315]

numOfObjects = 0
listOfUniqueIds = []

while True:     # This while function loops through all the frames of the input video and places labeled bounding box around objects of interest
    success, img = cap.read()

    imgRegion = cv.bitwise_and(img, mask)  # Combining the mask and the video
    # print(f"Shapes of images: {img.shape}")
    results = model(imgRegion, stream=True)   # running the model

    detections = np.empty((0, 5))   # Array of detections with attributes x1, y1, x2, y2 and the tracking ID

    for r in results:  # looping through results of a frame
        boxes = r.boxes                     # the r.boxes is used to extract the boxes from the tensor object 'r'
        # print(f"Here are boxes from one frame: {boxes}")
        for box in boxes:  # Looping through bounding boxes in the image 'r'
            # Using the CV2 package to draw bounding boxes
            # x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1),int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Using the CVZONE package to draw bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]  # The index '0' is meant to extract the data from the box.xyxy tensor object
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox = x1, y1, x2, y2
            w, h = x2-x1, y2-y1
            print(x1, y1, w, h)
            # cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=10, t=2)

            # Displaying the object class and confidence value using CVZONE
            conf = math.ceil(box.conf*100)/100
            print(f"Confidence = {conf}")
            class_index = box.cls
            objectClass = classNames[int(class_index[0])]
            print(f"Class Name = {objectClass}")

            # Selecting which object to detect (Object of interest for detection)
            if objectClass == 'car' or objectClass == 'bus' or objectClass == 'motorbike' or objectClass == 'truck'\
                and conf > 0.3:
                # cvzone.putTextRect(img,
                #                    f'{objectClass} {conf}',
                #                    (max(0, x1), max(35, y1)),
                #                    scale=1, thickness=1,
                #                    offset=3)  # This line prints the label with its confidence on the corner rectangle
                # cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=10, t=5)  # Drawing the bounding box
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = numpy.vstack((detections, currentArray))  # Appending arrays of detections with numpy

    # Performing the object tracking

    cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
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
        cx, cy = int(x1+w/2), int(y1+h/2)
        cv.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Setting the counting area in the image and counting the number of objects while tracking them

        # print(f"Ids = {resultTracker[:, -1]}")  # Printing all the IDs extracted from a single frame
        if limits[0]< cx < limits[2] and limits[1]-20 < cy < limits[1]+20: # For all the objects in this area, if the
            # center of an object is in this area of the image then count the object
            detectedIds = resultTracker[:, -1]
            for ID in resultTracker[:, -1]:
                if ID in listOfUniqueIds:
                    continue
                else:
                    listOfUniqueIds.append(ID)
                    numOfObjects += 1
                    cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
        cvzone.putTextRect(img, f'Count={numOfObjects}', (50, 50))  # Printing the number of objects detected

    cv.imshow("Image", img)
    # cv.imshow("ImageRegion", imgRegion)
    cv2.waitKey(0)
