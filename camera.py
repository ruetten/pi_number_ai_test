from picamera2 import Picamera2

picam2 = Picamera2()
picam2.start_preview()
picam2.start()

input("Press Enter to stop...")

picam2.stop()