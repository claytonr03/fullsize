# TODO:
# Create CalibratedCamera object
#   - Calibrate camera intrinsics
#   - Calibrate pixel size 

import cv2
import picamera
# import picamera.array
import io
import json

ASYMMETRIC_CIRCLES_SHAPE = (4, 11)

class CalibratedPiCamera:
  scale = (1, 1)
  type = "none"
  cam = None
  cal_data = {}

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
  def capture_calibrated(self):
    raw_image = self.capture_raw()

    pass 

  # Calibration function to determine camera intrinsics
  def calibrate_intrinsics(self):
    pass

  # Calibration function to determine camera scaling 
  def calibrate_scale(self):
    pass

  def calibrate(self):
    self.calibrate_intrinsics()
    self.calibrate_scale()

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

