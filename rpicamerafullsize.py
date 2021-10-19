import cv2 
import numpy as np
import os
import time
import json

from gpiozero import LED

image_directory = '/home/pi/Documents/captures'

ledBase = LED(27)
ledTop = LED(17)
ledBase.off()
ledTop.off()
time.sleep(.1)
ledTop.on()
ledBase.on()
time.sleep(.1)


def correct_image(calibration_filepath, image_filepath):
  with open(calibration_filepath) as cal_f:
    cal_data = dict(json.load(cal_f).items())
  print(cal_data)
  camera_matrix = np.array(cal_data["camera_matrix"])
  distortion_coefficient = np.array(cal_data["distortion_coefficient"])
  raw_image = cv2.imread(image_filepath)
  h, w = raw_image.shape[:2]
  newcamera, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficient,(w, h), 1, (w,h))
  corrected_image = cv2.undistort(raw_image, camera_matrix, distortion_coefficient, None, newcamera)
  return corrected_image

if __name__ == "__main__":
	
  # cv2.namedWindow("Raw Image")
  # cv2.namedWindow("Corrected Image")
  # raw_image = cv2.imread("./sample_image.jpg")
  # corrected_image = correct_image("./camera_calibration_data.json", "./sample_image.jpg")
  # cv2.imshow("Raw Image", raw_image)
  # cv2.imshow("Corrected Image", corrected_image)

  ledTop.off()
	#print("Top On before photo")
  time.sleep(1)
  os.system('raspistill -t 3000 -o {}/photo_1.jpg'.format(image_directory))
  ledTop.on()
  print("Photo 1 taken")

  # Correct photo 1:
  p1_corrected = correct_image('./camera_calibration_data.json', '{}/photo_1.jpg'.format(image_directory))
  cv2.imwrite('{}/photo_1_corrected.jpg'.format(image_directory), p1_corrected)

  time.sleep(.1)
  ledBase.off()
	#print("base on taking photo")
  time.sleep(1)
  os.system('raspistill -t 3000 -o {}/photo_2.jpg'.format(image_directory))
  ledBase.on()
  print("Photo 2 taken")

  # Correct photo 2:
  p2_corrected = correct_image('./camera_calibration_data.json', '{}/photo_2.jpg'.format(image_directory))
  cv2.imwrite('{}/photo_2_corrected.jpg'.format(image_directory), p2_corrected)


  time.sleep(.1)