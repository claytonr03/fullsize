# TODO:
# Complete camera intrinsics calibration function
# Integrate returned scaled images with svg code



import cv2
import picamera
# import picamera.array


from imutils import perspective
from imutils import contours
import imutils 

import numpy as np

import io
import json


ASYMMETRIC_CIRCLES_SHAPE = (4, 11)

class CalibratedPiCamera:
  scale = (1, 1)
  type = "none"
  cam = None
  cal_data = {}

  pixels_per_metric_x = None
  pixels_per_metric_y = None

  def __init__(self, type, calibration_filepath):
    self.type = type

    with open(calibration_filepath) as cal_f:
      self.cal_data = dict(json.load(cal_f).items())

    print(self.cal_data)

    # picam specific setup
    if self.type == "pi":
      self.cam = picamera.PiCamera()

    # webcam specific setup
    if self.type == "webcam":
      self.cam = cv2.VideoCapture(0)

  # Capture and return uncalibrated image
  def capture_raw(self):
    if self.type == "pi":
      raw_capture = picamera.array.PiRGBArray(self.cam)
      self.cam.capture(raw_capture, format="bgr")
      return raw_capture.array
    
    if self.type == "webcam":
      for retries in range(0, 10):
        ret, raw_capture = self.cam.read()
        if ret:
          return raw_capture
    
    return None

  # Capture and return calibrated image
  # TODO: Note - temporary return raw_image
  def capture_calibrated(self):
    raw_image = self.capture_raw()
    return raw_image

  # Calibration function to determine camera intrinsics
  # TODO: Complete intrinsics calibration function 
  def calibrate_intrinsics(self):
    raw_image = self.capture_raw()
    cv2.namedWindow('intrinsics calibration')
    blobs = self.find_blobs(raw_image)
    # print(blobs)
    if blobs.any():
      drawn_image = cv2.drawChessboardCorners(raw_image, ASYMMETRIC_CIRCLES_SHAPE, blobs, True)
      cv2.imshow('intrinsics calibration', drawn_image)
    else:
      cv2.imshow('intrinsics calibration', raw_image)
    cv2.waitKey(0)

  # Calibration function to determine camera scaling 
  def calibrate_scale(self, object_diameter):
    print("Place the Calibration Grid into the center of the view area")

    cv2.namedWindow('scale calibration')
    calib_image = self.capture_calibrated()
    calib_image_gray = cv2.cvtColor(calib_image, cv2.COLOR_BGR2GRAY)
    calib_image_gray = cv2.GaussianBlur(calib_image_gray, (7, 7), 0)

    edged = cv2.Canny(calib_image_gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cntrs = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = imutils.grab_contours(cntrs)

    (cntrs, _) = contours.sort_contours(cntrs)

    drawn_image = calib_image.copy()
    count = 0
    (total_pixel_width, total_pixel_height) = (0,0)
    for c in cntrs:
      if cv2.contourArea(c) < 100:
        continue
    
      
      box = cv2.minAreaRect(c)
      total_pixel_width += box[1][0]
      total_pixel_height += box[1][1]
      points = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
      points = np.array(points, dtype="int")
      # order the points in the contour such that they appear
      # in top-left, top-right, bottom-right, and bottom-left
      # order, then draw the outline of the rotated bounding
      # box
      points = perspective.order_points(points)
      drawn_image = cv2.drawContours(drawn_image, [points.astype("int")], -1, (0, 255, 0), 2)
      
      count += 1

    cv2.imshow('scale calibration', drawn_image)
    cv2.waitKey(0)

    width_avg = (total_pixel_width / count)
    height_avg = (total_pixel_height / count)

    print("Detected {} calibration object(s)".format(count))
    print("Avg Width: {}px, Avg Height: {}px".format(width_avg, height_avg))

    # if pixels_per_metric_x is None:
    #   pixels_per_metric_x = width_avg/object_radius
    # if pixels_per_metric_y is None:
    #   pixels_per_metric_y = height_avg/object_radius

    self.pixels_per_metric_x = width_avg/object_diameter
    self.pixels_per_metric_y = height_avg/object_diameter

  def calibrate(self, cal_object_diameter=None):
    if cal_object_diameter is None:
      cal_object_diameter = float(input("Enter calibration dot diameter: "))
    self.calibrate_intrinsics()
    self.calibrate_scale(cal_object_diameter)

  def print_calibration_data(self):
    print("\n======================")
    print("Calibration Data")
    print("======================")
    print("\n----------------------")
    print("Intrinsics Calibration")
    print("----------------------")
    print("Camera Matrix: {}".format(self.cal_data['camera_matrix']))
    print("Distortion Coefficient: {}".format(self.cal_data['distortion_coefficient']))
    print("\n----------------------")
    print("Scale Calibration")
    print("----------------------")
    print("X: {} px/unit".format(self.pixels_per_metric_x))
    print("Y: {} px/unit".format(self.pixels_per_metric_y))
    print("======================")

  def correct_image(calibration_filepath, image_filepath):
    camera_matrix = np.array(cal_data["camera_matrix"])
    distortion_coefficient = np.array(cal_data["distortion_coefficient"])
    raw_image = cv2.imread(image_filepath)
    h, w = raw_image.shape[:2]
    newcamera, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficient,(w, h), 1, (w,h))
    corrected_image = cv2.undistort(raw_image, camera_matrix, distortion_coefficient, None, newcamera)
    return corrected_image

  def store_raw_capture(self, filename):
    raw_image = self.capture_raw()
    cv2.imwrite(filename, raw_image)


  def display_raw_capture(self):
    raw_image = self.capture_raw()
    cv2.namedWindow('raw capture')
    cv2.imshow('raw capture', raw_image)
    cv2.waitKey(0)

  def find_blobs(self, image):
    ret, points = cv2.findCirclesGrid(image, ASYMMETRIC_CIRCLES_SHAPE, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    if ret:
      return points
    return None

  def display_blobs(self):
    raw_image = self.capture_raw()
    cv2.namedWindow('blobs')
    blobs = self.find_blobs(raw_image)
    # print(blobs)
    if blobs.any():
      drawn_image = cv2.drawChessboardCorners(raw_image, ASYMMETRIC_CIRCLES_SHAPE, blobs, True)
      cv2.imshow('blobs', drawn_image)
    else:
      cv2.imshow('blobs', raw_image)
    cv2.waitKey(0)