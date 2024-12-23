import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

####################################
wCam, hCam = 640, 480
####################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.HandDetector(detection_con=0.5)

# Initialize audio interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()  # Get volume range [-65, 0]
minVol = volRange[0]
maxVol = volRange[1]

# Volume variables
vol = 0
volBar = 400
volPer = 0
scale_factor = 1.008  # adjust this slightly up or down for sensitivity

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Thumb and index finger coordinates
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Visual feedback: draw circles and lines between fingers
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # Calculate distance between thumb and index finger
        length = math.hypot(x2 - x1, y2 - y1)

        # Interpolate volume based on log of the distance
        vol = np.interp(np.log(length) * scale_factor, [np.log(50), np.log(300)], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])

        # Ensure volume stays within range
        vol = np.clip(vol, minVol, maxVol)
        print(f"Volume Level: {vol}")

        # Set the system volume
        volume.SetMasterVolumeLevel(vol, None)

        # Change color when volume control gesture is near minimum distance
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # Draw the volume bar and percentage text
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f"{int(volPer)}%", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

    # Frame rate calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS
    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the image
    cv2.imshow("Img", img)

    # Use a small delay to reduce CPU usage and smooth performance
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release video capture object
cap.release()
cv2.destroyAllWindows()
