
# For x86 development:
# import sys
# import fake_rpi
# sys.modules['RPi'] = fake_rpi.RPi     # Fake RPi
# sys.modules['picamera'] = fake_rpi.picamera # Fake picamera
# sys.modules['RPi.GPIO'] = fake_rpi.RPi.GPIO



from calibratedcamera import CalibratedPiCamera
import cv2 
import numpy as np
import os
import time
import json

import RPi.GPIO as GPIO
#GPIO.setmode(GPIO.BOARD)
GPIO.setmode(GPIO.BCM)
# ledBase = LED(27)
# ledTop = LED(17)
# ledBase.off()
# ledTop.off()
# time.sleep(.1)
# ledTop.on()
# ledBase.on()
# time.sleep(.1)

class ImagingStand:

  led_base = None
  led_top = None
  cam = None
  display_size = (640, 480)

  def __init__(self, led_top_pin, led_base_pin, camera_type="pi", calibration_file=None):
    self.led_top = led_top_pin
    self.led_base = led_base_pin
    self.init_gpio()

    # GPIO are inverted
    # TODO: fix inversion
    self.set_led(self.led_top, 1)
    self.set_led(self.led_base, 1)

    # Initialize camera:
    self.cam = CalibratedPiCamera(camera_type, calibration_file)


  def init_gpio(self):
    GPIO.setup(self.led_top, GPIO.OUT)
    GPIO.setup(self.led_base, GPIO.OUT)

  def set_led(self, led, value):
    GPIO.output(led, value)
  
  def capture_unlit(self):
    self.set_led(self.led_top, 1)
    self.set_led(self.led_base, 1)
    return self.cam.capture_calibrated()

  def calibrate_all(self):
    # GPIO are inverted
    self.set_led(self.led_top, 0)
    self.cam.calibrate()
    self.set_led(self.led_top, 1)

  def calibrate_intrinsics(self):
    x = int(input("Enter pattern shape X: "))
    y = int(input("Enter pattern shape Y: "))
    pattern_shape = (x,y)
    self.set_led(self.led_top, 0)
    self.cam.calibrate_intrinsics(pattern_shape)
    self.set_led(self.led_top, 1)

  def calibrate_scale(self):
    cal_object_diameter = float(input("Enter calibration dot diameter: "))
    cal_object_units = input("Enter calibration dot measurement units (in/mm): ")
    self.set_led(self.led_top, 0)
    # Calibration routines:
    self.cam.calibrate_scale(cal_object_diameter, cal_object_units)
    self.cam.save_intrinsics()
    self.set_led(self.led_top, 1)

  def capture_top(self):
    # GPIO are inverted
    self.set_led(self.led_top, 0)
    image = self.cam.capture_calibrated()
    self.set_led(self.led_top, 1)
    return image

  def capture_bottom(self):
    # GPIO are inverted
    self.set_led(self.led_base, 0)
    image = self.cam.capture_calibrated()
    self.set_led(self.led_base, 1)
    return image

  def get_pixel_metrics(self):
    return self.cam.get_pixel_metrics()




if __name__ == "__main__":


  is1 = ImagingStand(17, 27, "webcam", "./camera_calibration_data_generated.json")

  cv2.namedWindow('Capture')
  image = is1.capture_unlit()
  cv2.imshow('Capture', image)
  cv2.waitKey(0)

  image = is1.capture_top()
  cv2.imshow('Capture', image)
  cv2.waitKey(0)
  
  image = is1.capture_bottom()
  cv2.imshow('Capture', image)
  cv2.waitKey(0)
  
  
  
  # c1 = CalibratedPiCamera("webcam", "./camera_calibration_data.json")
  # # c1.calibrate()
  # c1.load_calibration_file("./camera_calibration_data_generated.json")
  # c1.print_calibration_data()

  # cv2.namedWindow('Raw Capture')
  # image_raw = c1.capture_raw()
  # cv2.imshow('Raw Capture', image_raw)

  # cv2.namedWindow('Calibrated Capture')
  # image = c1.capture_calibrated()
  # cv2.imshow('Calibrated Capture', image)

  # cv2.waitKey(0)

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
