from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.start()

while True:
    cv2.imshow("Camera", picam2.capture_array())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()