
# For x86 development:
import sys
import fake_rpi
sys.modules['RPi'] = fake_rpi.RPi     # Fake RPi
sys.modules['picamera'] = fake_rpi.picamera # Fake picamera



from calibratedcamera import CalibratedPiCamera
import cv2 
import numpy as np
import os
import time
import json

# from gpiozero import LED

image_directory = '/home/pi/Documents/captures'



# ledBase = LED(27)
# ledTop = LED(17)
# ledBase.off()
# ledTop.off()
# time.sleep(.1)
# ledTop.on()
# ledBase.on()
# time.sleep(.1)


if __name__ == "__main__":
  c1 = CalibratedPiCamera("webcam", "./camera_calibration_data.json")
  # c1.display_raw_capture()
  # c1.display_blobs()
  c1.circle_scale_calibration(24.257)


  # c2 = CalibratedPiCamera("pi", "./camera_calibration_data.json")
	
  # cv2.namedWindow("Raw Image")
  # cv2.namedWindow("Corrected Image")
  # raw_image = cv2.imread("./sample_image.jpg")
  # corrected_image = correct_image("./camera_calibration_data.json", "./sample_image.jpg")
  # cv2.imshow("Raw Image", raw_image)
  # cv2.imshow("Corrected Image", corrected_image)

  # ledTop.off()
	# #print("Top On before photo")
  # time.sleep(1)
  # os.system('raspistill -t 3000 -o {}/photo_1.jpg'.format(image_directory))
  # ledTop.on()
  # print("Photo 1 taken")

  # # Correct photo 1:
  # p1_corrected = correct_image('./camera_calibration_data.json', '{}/photo_1.jpg'.format(image_directory))
  # cv2.imwrite('{}/photo_1_corrected.jpg'.format(image_directory), p1_corrected)

  # time.sleep(.1)
  # ledBase.off()
	# #print("base on taking photo")
  # time.sleep(1)
  # os.system('raspistill -t 3000 -o {}/photo_2.jpg'.format(image_directory))
  # ledBase.on()
  # print("Photo 2 taken")

  # # Correct photo 2:
  # p2_corrected = correct_image('./camera_calibration_data.json', '{}/photo_2.jpg'.format(image_directory))
  # cv2.imwrite('{}/photo_2_corrected.jpg'.format(image_directory), p2_corrected)


  # time.sleep(.1)