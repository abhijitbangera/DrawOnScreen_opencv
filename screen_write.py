import cv2
from collections import deque
import numpy as np

Lower_black = np.array([20, 100, 100])
Upper_black = np.array([30, 255, 255])

cap = cv2.VideoCapture(0)
pts = deque(maxlen=512)
blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
while True:
    z, img = cap.read()
    # Mirror image
    img = cv2.flip(img, 1)
    # Extracting the contours from the image
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, Lower_black, Upper_black)
    blur = cv2.medianBlur(mask, 15)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

    center = None
    if len(cnts) >= 1:
        contour = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(contour) > 250:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(img, center, 5, (0, 0, 255), -1)
            M = cv2.moments(contour)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            pts.appendleft(center)
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 10)
                cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 5)
    elif len(cnts) == 0:
        if len(pts) != []:
            blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
            blur1 = cv2.medianBlur(blackboard_gray, 15)
            blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
            thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
            if len(blackboard_cnts) >= 1:
                cnt = max(blackboard_cnts, key=cv2.contourArea)
                print(cv2.contourArea(cnt))
                if cv2.contourArea(cnt) > 2000:
                    x, y, w, h = cv2.boundingRect(cnt)
        pts = deque(maxlen=512)
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imshow("Frame", img)
    k = cv2.waitKey(1)
    if k == 27:
        break


