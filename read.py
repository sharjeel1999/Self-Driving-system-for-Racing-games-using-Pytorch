import cv2
import numpy as np

cap = cv2.VideoCapture(2)

while True:
    _, frame1 = cap.read()
    _, frame2 = cap.read()
    
    d = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)    
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3,3), np.uint8), iterations=3)
    
    cv2.imshow("Live", dilated)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()