import cv2
import mediapipe as mp
from math import hypot
import numpy as np
import os

cap = cv2.VideoCapture(0)  # Checks for camera

mpHands = mp.solutions.hands  # Detects hand/finger
hands = mpHands.Hands()  # Complete the initialization configuration of hands
mpDraw = mp.solutions.drawing_utils

volbar = 400
volper = 0

while True:
    success, img = cap.read()  # If camera works capture an image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to rgb

    # Collection of gesture information
    results = hands.process(imgRGB)  # Completes the image processing.

    lmList = []  # Empty list
    if results.multi_hand_landmarks:  # List of all hands detected.
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):  # Adding counter and returning it
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])  # Adding to the empty list 'lmList'
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

    if lmList:
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger
        cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = hypot(x2 - x1, y2 - y1)  # Distance between tips using hypotenuse

        # Interpolate hand distance into volume percentage (0 to 100)
        volper = np.interp(length, [30, 350], [0, 100])

        # Set volume using AppleScript
        script = f'''
        osascript -e "set volume output volume {volper}"
        '''
        os.system(script)

        # Displaying the volume bar
        volbar = np.interp(length, [30, 350], [400, 150])
        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
        cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)

    cv2.imshow('Image', img)  # Show the video
    if cv2.waitKey(1) & 0xff == ord(' '):  # Using spacebar delay will stop
        break

cap.release()  # Stop cam
cv2.destroyAllWindows()  # Close window
