# TODO:
# Complete camera intrinsics calibration function
# Integrate returned scaled images with svg code



import cv2
import picamera
import picamera.array


from imutils import perspective
from imutils import contours
import imutils 

import numpy as np

import io
import json

import time

class CalibratedPiCamera:
  scale = (1, 1)
  type = "none"
  cam = None
  cal_data = {}
  num_cal_images = 12

  def __init__(self, type, calibration_filepath=None):
    self.type = type

    self.load_calibration_file(calibration_filepath)
    print(self.cal_data)

    # picam specific setup
    if self.type == "pi":
      self.cam = picamera.PiCamera(sensor_mode=2)
      self.cam.resolution = (2592, 1944)

    # webcam specific setup
    if self.type == "webcam":
      self.cam = cv2.VideoCapture(0)

  def load_calibration_file(self, calibration_filepath):
    with open(calibration_filepath) as cal_f:
      self.cal_data = dict(json.load(cal_f).items())

  # Capture and return uncalibrated image
  def capture_raw(self):
    if self.type == "pi":
      raw_capture = picamera.array.PiRGBArray(self.cam)
      self.cam.capture(raw_capture, format="bgr")
      return raw_capture.array
    
    if self.type == "webcam":
      for retries in range(0, 10):
        # hacky workaround to update the webcam image - not sure why it doesn't always update
        for i in range(0, 10):
          ret, raw_capture = self.cam.read()
        if ret:
          return raw_capture
    
    return None

  # Capture and return calibrated image cropped to the ROI
  def capture_calibrated(self):
    raw_image = self.capture_raw()

    h, w = raw_image.shape[:2]
    matrix = np.array(self.cal_data['camera_matrix'])
    distortion = np.array(self.cal_data['distortion_coefficient'])
    newcamera, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion,(w, h), 1, (w,h))
    undist_image = cv2.undistort(raw_image, matrix, distortion, None, newcamera)
    
    print("ROI: {}".format(roi))

    x, y, w, h = roi
    # hacky workaround to remove edges until proper calibration ROI method is implemented
    margin_x = int(.1 * w)
    margin_y = int(.1 * h)
    undist_image = undist_image[y+margin_y:y+h-margin_y, x+margin_x:x+w-margin_x]

    return undist_image

  # Calibration function to determine camera intrinsics
  # TODO: Complete intrinsics calibration function 
  def calibrate_intrinsics(self, pattern_shape):
    criteria = (cv2.TERM_CRITERIA_EPS +
              cv2.TERM_CRITERIA_MAX_ITER, 10, 1)


    # Vector for 3D points
    threedpoints = []
    
    # Vector for 2D points
    twodpoints = []
    
    objectp3d = np.zeros((1, pattern_shape[0]
                          * pattern_shape[1],
                          3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:pattern_shape[0],
                                  0:pattern_shape[1]].T.reshape(-1, 2)


    cv2.namedWindow('intrinsics calibration')
    
     
    for i in range(0, self.num_cal_images):
      while True:
        raw_image = self.capture_raw()
        gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
         
        #max_value = 255
        #neighborhood = 99
        #subtract_from_mean = 30
        #gray_image = cv2.adaptiveThreshold(gray_image,
        #        max_value,
        #        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #        cv2.THRESH_BINARY,
        #        neighborhood,
        #        subtract_from_mean)
        
        #gray_image = cv2.gaussian
        #gray_image = cv2.erode(gray_image, None, iterations=1)

        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        # corners = self.find_blobs(gray_image)
        #corners = self.find_blobs(gray_image, pattern_shape)

        corners = self.find_chessboard(gray_image, pattern_shape)
    
        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if corners is not None:
            threedpoints.append(objectp3d)
    
            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(
                gray_image, corners, (11, 11), (-1, -1), criteria)
            
            #corners2 = corners

            twodpoints.append(corners2)

            drawn_image = cv2.drawChessboardCorners(raw_image,
                                              pattern_shape,
                                              corners2, True)

            drawn_image_preview = drawn_image.copy()
            drawn_image_preview = cv2.resize(drawn_image_preview, (800, 600))
            cv2.imshow('intrinsics calibration', drawn_image_preview)
            break

        else:
          print("Could not find calibration pattern, please re-align and press any key to retry")
         
          gray_image_preview = gray_image.copy()
          gray_image_preview = cv2.resize(gray_image_preview, (800, 600))
          cv2.imshow('intrinsics calibration', gray_image_preview)
          cv2.waitKey(0) 
        
      cv2.waitKey(0)  

    h, w = drawn_image.shape[:2]
    

    # Perform camera calibration by
    # passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the
    # detected corners (twodpoints)
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, gray_image.shape[::-1], None, None)
    self.cal_data['camera_matrix'] = matrix.tolist()
    self.cal_data['distortion_coefficient'] = distortion.tolist()
    cv2.destroyWindow('intrinsics calibration')


  # raw_image = self.capture_raw()
  # cv2.namedWindow('intrinsics calibration')
  # blobs = self.find_blobs(raw_image)
  # # print(blobs)
  # if blobs.any():
  #   drawn_image = cv2.drawChessboardCorners(raw_image, ASYMMETRIC_CIRCLES_SHAPE, blobs, True)
  #   cv2.imshow('intrinsics calibration', drawn_image)
  # else:
  #   cv2.imshow('intrinsics calibration', raw_image)
  # cv2.waitKey(0)

  # TODO: save intrinsics to file 
  def save_intrinsics(self):
    with open("camera_calibration_data_generated.json", 'w') as f:
      json.dump(self.cal_data, f)


  def undistort_image(self, dist_image):
    h, w = dist_image.shape[:2]
    self.newcamera, roi = cv2.getOptimalNewCameraMatrix(self.matrix, self.distortion,(w, h), 1, (w,h))
    undist_image = cv2.undistort(dist_image, self.matrix, self.distortion, None, self.newcamera)
    return undist_image

  # Calibration function to determine camera scaling 
  def calibrate_scale(self, object_diameter, area_criteria):
    input("Place the Scale Calibration Grid into the center of the view area. (Press Enter to continue)")

    cv2.namedWindow('scale calibration')
    calib_image = self.capture_calibrated()
    calib_image_gray = cv2.cvtColor(calib_image, cv2.COLOR_BGR2GRAY)

    #calib_image_gray = cv2.GaussianBlur(calib_image_gray, (7, 7), 0)
    
    max_output = 255
    subtract_from_mean = 40
    neighborhood = 99
    calib_image_gray = cv2.adaptiveThreshold(calib_image_gray,
            max_output,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            neighborhood,
            subtract_from_mean)
            
    #cv2.imshow('scale calibration', calib_image_gray)
    #cv2.waitKey(0)

    edged = cv2.Canny(calib_image_gray, 50, 100)
    #cv2.imshow('scale calibration', edged)
    #cv2.waitKey(0)

    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    #cv2.imshow('scale calibration', edged)
    #cv2.waitKey(0)

    cntrs = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = imutils.grab_contours(cntrs)

    (cntrs, _) = contours.sort_contours(cntrs)

    drawn_image = calib_image.copy()
    count = 0
    (total_pixel_width, total_pixel_height) = (0,0)

    # TODO: Need better contour rejection criteria:
    for c in cntrs:
      if cv2.contourArea(c) < area_criteria:
        continue

      x,y,w,h = cv2.boundingRect(c)
      total_pixel_width += w
      total_pixel_height += h
      drawn_image = cv2.rectangle(drawn_image, (x,y), (x+w, y+h), (0,255,0), 15)

      drawn_image_preview = drawn_image.copy()
      drawn_image_preview = cv2.resize(drawn_image_preview, (640, 480))
      count += 1

    cv2.imshow('scale calibration', drawn_image_preview)
    cv2.waitKey(0)

    # Note: Do the rotation values of the box potentially impact the calibration accuracy?
    width_avg = (total_pixel_width / count)
    height_avg = (total_pixel_height / count)

    print("Detected {} calibration object(s)".format(count))
    print("Avg Width: {}px, Avg Height: {}px".format(width_avg, height_avg))

    # if pixels_per_metric_x is None:
    #   pixels_per_metric_x = width_avg/object_radius
    # if pixels_per_metric_y is None:
    #   pixels_per_metric_y = height_avg/object_radius

    pixels_per_metric_x = width_avg/object_diameter
    pixels_per_metric_y = height_avg/object_diameter
    self.cal_data['pixels_per_metric'] = [pixels_per_metric_x, pixels_per_metric_y]

  def calibrate(self, pattern_shape=None, cal_object_diameter=None, area_criteria=None):
    if pattern_shape is None:
      x = int(input("Enter pattern shape X: "))
      y = int(input("Enter pattern shape y: "))
      pattern_shape = (x,y)

    if cal_object_diameter is None:
      cal_object_diameter = float(input("Enter calibration dot diameter: "))

    if area_criteria is None:
        area_criteria = int(input("Enter the estimated dot diameter in pixels: "))

    # Calibration routines:
    self.calibrate_intrinsics(pattern_shape)
    self.calibrate_scale(cal_object_diameter, area_criteria)
    self.save_intrinsics()
    
  def get_pixel_metrics(self):
      return(self.cal_data['pixels_per_metric'])

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
    print("X: {} px/unit".format(self.cal_data['pixels_per_metric'][0]))
    print("Y: {} px/unit".format(self.cal_data['pixels_per_metric'][1]))
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

  def find_blobs(self, image, pattern_shape):
    blob_params = cv2.SimpleBlobDetector_Params()
    
    blob_params.minThreshold = 8
    blob_params.maxThreshold = 255

    blob_params.filterByArea = True
    blob_params.minArea = 64
    blob_params.maxArea = 3000

    blob_params.filterByCircularity = True
    blob_params.minCircularity = 0.1

    blob_detector = cv2.SimpleBlobDetector_create(blob_params)

    keypoints = blob_detector.detect(image)

    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    #cv2.namedWindow('test')
    cv2.imshow('intrinsics calibration', im_with_keypoints)

    #cv2.waitKey(0)
    time.sleep(1)

    #image = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)

    ret, points = cv2.findCirclesGrid(image, pattern_shape, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
    if ret:
      return points
    return None

  def find_chessboard(self, image, pattern_shape):
    ret, points = cv2.findChessboardCorners(image, pattern_shape,
                        cv2.CALIB_CB_ADAPTIVE_THRESH
                        + cv2.CALIB_CB_FAST_CHECK +
                        cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
      return points
    return None

  def display_blobs(self, pattern_shape):
    raw_image = self.capture_raw()
    cv2.namedWindow('blobs')
    blobs = self.find_blobs(raw_image, pattern_shape)
    # print(blobs)
    if blobs.any():
      drawn_image = cv2.drawChessboardCorners(raw_image, pattern_shape, blobs, True)
      cv2.imshow('blobs', drawn_image)
    else:
      cv2.imshow('blobs', raw_image)
    cv2.waitKey(0)
