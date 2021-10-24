import numpy as np
import imutils
import time
import cv2
import os
import email, smtplib, ssl
import sys

from imagingstand import ImagingStand

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Imports PIL module  
from PIL import Image 

class ContourMaker:

  imager = None

  def __init__(self):
    self.imager = ImagingStand(17, 27, "webcam", "./camera_calibration_data_generated.json")


  def auto_canny(self, image, sigma=0.33):

    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged
  
  def generate_contour_svg(self, filepath, toolnumber):
    cv2.namedWindow('generate_contour_svg')

    # -------------------------
    # Capture Top Image
    # -------------------------
    frame = self.imager.capture_top()
    scale_percentx = 69.5
    scale_percenty = 69.5
    width = int(frame.shape[1] * scale_percentx / 100)
    height = int(frame.shape[0] * scale_percenty / 100)
    dsize = (width, height)
    frame = cv2.resize(frame, dsize)
    #Y is first with Yo and Y distance
    #frameT = frame[400:2600, 1000:3300]
    # frameT = frame[120:4000, 550:3500]

    cv2.imshow('generate_contour_svg', frame)
    cv2.waitKey(0)
    
    # ------------------------
    # Capture Bottom Image
    # ------------------------
    frame = self.imager.capture_bottom()
    #scale_percentx = 69.5
    #scale_percenty = 69.5
    width = int(frame.shape[1] * scale_percentx / 100)
    height = int(frame.shape[0] * scale_percenty / 100)
    dsize = (width, height)
    #frame = cv2.resize(frame, ds)ize)
    # frame = frame[120:4000, 550:3500]
    
    cv2.imshow('generate_contour_svg', frame)
    cv2.waitKey(0)

    # ------------------------
    # Generate Contours
    # ------------------------
    fgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(fgGray, 0, 255,
      cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #cv2.namedWindow("test",cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("test",cv2.WND_PROP_FULLSCREEN,cv2.CV_WINDOW_FULLSCREEN)

    thresh = cv2.erode(thresh, None, iterations=5)#4
    thresh = cv2.dilate(thresh, None, iterations=1)#4
    cv2.imshow('generate_contour_svg', thresh)
    cv2.waitKey(0)

    blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
    edged = cv2.Canny(thresh, 50, 130)
    edged = cm1.auto_canny(thresh)

    cv2.imshow('generate_contour_svg', edged)
    cv2.waitKey(0)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # ------------------------
    # Generate SVG
    # ------------------------
    total = 0
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #print(timestr)

    c = max(cnts, key=cv2.contourArea) #max contour
    f = open(filepath+'/'+toolnumber+'_'+timestr+'_vectors.svg', 'w+')
    f.write('<svg width="4000" height="4000" xmlns="http://www.w3.org/2000/svg">')
    # loop over the contours one by one
    for c in cnts:
      if cv2.contourArea(c) < 500:	#or cv2.contourArea(c) > 200:
        continue
    #	print(" contour {}".format(cv2.contourArea(c)))
      x,y,w,h = cv2.boundingRect(c)
    #	print(x)
    #	print(y)
    #	print(w)
    #	print(h)
      cv2.drawContours(frame, [c], -1, (204, 0, 255), 2)
      total += 1
      f.write('<path d="M')
      for i in range(len(c)):
        if i == 0:
          x1, y1 = c[i][0]
        #print(c[i][0])
        x, y = c[i][0]
        #print(x)
        f.write(str(x)+  ' ' + str(y)+' ')
      f.write(str(x1)+  ' ' + str(y1)+' ')
      f.write('"/>')

    f.write('</svg>')
    f.close()


if __name__ == "__main__":
  cm1 = ContourMaker()
  cm1.generate_contour_svg("./", "1")

exit(1)

if len(sys.argv) > 1:
 usb = sys.argv[1]
else:
 usb = "-l"

if usb == "-u":
 print("All files will be saved in directory /media/pi/KINGSTON (there needs to be a KINGSTON USB inserted)")
 filePath = "/media/pi/KINGSTON"
elif usb == "-h":
 print("Use the -u switch to make the system save to the USB drive /media/pi/KINGSTON. Otherwise it saves in /home/pi/Documents/betterprocess/toolimages. Use the -e switch to send email.")
 sys.exit()
elif usb == "-e":
 print("All files will be sent via email and all files will be saved in directory /home/pi/Documents/betterprocess/toolimages")
 filePath = "/home/pi/Documents/betterprocess/fullsize/toolimages"
elif usb == "-l":
 print("All files will be saved in directory /home/pi/Documents/betterprocess/fullsize/toolimages")
 filePath = "/home/pi/Documents/betterprocess/fullsize/toolimages"

z_dim = sys.argv[2]
toolnumber = sys.argv[3]
x_dim = sys.argv[4]
y_dim = sys.argv[5]

###########################
# frame = cv2.imread("/home/pi/Documents/unfish/corrected_images/TopView.jpg",1)
frame = cm1.imager.capture_top()
scale_percentx = 69.5
scale_percenty = 69.5
width = int(frame.shape[1] * scale_percentx / 100)
height = int(frame.shape[0] * scale_percenty / 100)
dsize = (width, height)
frame = cv2.resize(frame, dsize)
#Y is first with Yo and Y distance
#frameT = frame[400:2600, 1000:3300]
frameT = frame[120:4000, 550:3500]
###########################
 
# frame = cv2.imread("/home/pi/Documents/unfish/corrected_images/workingImage.jpg",1)
frame = cm1.imager.capture_bottom()
#scale_percentx = 69.5
#scale_percenty = 69.5
width = int(frame.shape[1] * scale_percentx / 100)
height = int(frame.shape[0] * scale_percenty / 100)
dsize = (width, height)
#frame = cv2.resize(frame, ds)ize)
frame = frame[120:4000, 550:3500]

#print(dsize)

#Y is first with Yo and Y distance
#frame = frame[400:2600, 1000:3300]

fgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(fgGray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#cv2.namedWindow("test",cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("test",cv2.WND_PROP_FULLSCREEN,cv2.CV_WINDOW_FULLSCREEN)

thresh = cv2.erode(thresh, None, iterations=5)#4
thresh = cv2.dilate(thresh, None, iterations=1)#4
cv2.imshow("test2", thresh)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
edged = cv2.Canny(thresh, 50, 130)
edged = cm1.auto_canny(thresh)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
total = 0
timestr = time.strftime("%Y%m%d-%H%M%S")
#print(timestr)

c = max(cnts, key=cv2.contourArea) #max contour
f = open(filePath+'/'+toolnumber+'_'+timestr+'_vectors.svg', 'w+')
f.write('<svg width="4000" height="4000" xmlns="http://www.w3.org/2000/svg">')


# loop over the contours one by one
for c in cnts:
	if cv2.contourArea(c) < 500:	#or cv2.contourArea(c) > 200:
		continue
#	print(" contour {}".format(cv2.contourArea(c)))
	x,y,w,h = cv2.boundingRect(c)
#	print(x)
#	print(y)
#	print(w)
#	print(h)
	cv2.drawContours(frame, [c], -1, (204, 0, 255), 2)
	total += 1
	f.write('<path d="M')
	for i in range(len(c)):
		if i == 0:
			x1, y1 = c[i][0]
		#print(c[i][0])
		x, y = c[i][0]
		#print(x)
		f.write(str(x)+  ' ' + str(y)+' ')
	f.write(str(x1)+  ' ' + str(y1)+' ')
	f.write('"/>')

f.write('</svg>')
f.close()

print("[INFO] found {} shapes".format(total))
#cv2.imshow("Debug2", frame)
cv2.imwrite(filePath+'/'+toolnumber+'_'+timestr+'_contours.jpg', frame)
cv2.imwrite(filePath+'/'+toolnumber+'_'+timestr+'_topView.jpg', frameT)
#cv2.imwrite('/media/pi/1.0 GB Volume/contours'+timestr+'.jpg', frame)

#cv2.waitKey(0)

filename = filePath+'/dims.txt'

if os.path.exists(filename):
    append_write = 'a' # append if already exists
else:
    append_write = 'w' # make a new file if not

dims = open(filename,append_write)
dims.write(toolnumber + '_' + timestr + ", " + x_dim + ", " + y_dim + ", " + z_dim + '\n')
dims.close()

os.system('xdg-open '+filePath+'/'+toolnumber+'_'+timestr+'_contours.jpg')

if usb == "-e":
	subject = "Another SVG from TPC"
	body = timestr + ", " + toolnumber + ", " + x_dim + ", " + y_dim + ", " + z_dim + '\n'
	sender_email = "betterprocess.rpi@gmail.com"  # Enter your address
	receiver_email = "ken.rayment@betterprocess.com"  # Enter receiver address
	password = "EB4wZhR3s5dxLi4"

	# Create a multipart message and set headers
	message = MIMEMultipart()
	message["From"] = sender_email
	message["To"] = receiver_email
	message["Subject"] = subject
	message["Bcc"] = receiver_email  # Recommended for mass emails

	# Add body to email
	message.attach(MIMEText(body, "plain"))

	filename = filePath+'/vectors'+timestr+'.svg'  # In same directory as script

	# Open PDF file in binary mode
	with open(filename, "rb") as attachment:
		# Add file as application/octet-stream
		# Email client can usually download this automatically as attachment
		part = MIMEBase("application", "octet-stream")
		part.set_payload(attachment.read())

	# Encode file in ASCII characters to send by email    
	encoders.encode_base64(part)

	# Add header as key/value pair to attachment part
	part.add_header(
		"Content-Disposition",
		f"attachment; filename= {filename}",
	)

	# Add attachment to message and convert message to string
	message.attach(part)
	text = message.as_string()

	#################################################

	filename = filePath+'/contours'+timestr+'.jpg'  # In same directory as script

	# Open PDF file in binary mode
	with open(filename, "rb") as attachment:
		# Add file as application/octet-stream
		# Email client can usually download this automatically as attachment
		part = MIMEBase("application", "octet-stream")
		part.set_payload(attachment.read())

	# Encode file in ASCII characters to send by email    
	encoders.encode_base64(part)

	# Add header as key/value pair to attachment part
	part.add_header(
		"Content-Disposition",
		f"attachment; filename= {filename}",
	)

	# Add attachment to message and convert message to string
	message.attach(part)
	text = message.as_string()

	####

	filename = filePath+'/dims.txt'  # In same directory as script

	# Open PDF file in binary mode
	with open(filename, "rb") as attachment:
		# Add file as application/octet-stream
		# Email client can usually download this automatically as attachment
		part = MIMEBase("application", "octet-stream")
		part.set_payload(attachment.read())

	# Encode file in ASCII characters to send by email    
	encoders.encode_base64(part)

	# Add header as key/value pair to attachment part
	part.add_header(
		"Content-Disposition",
		f"attachment; filename= {filename}",
	)

	# Add attachment to message and convert message to string
	message.attach(part)
	text = message.as_string()


	#################################################
	# Log in to server using secure context and send email
	context = ssl.create_default_context()
	with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
		server.login(sender_email, password)
		server.sendmail(sender_email, receiver_email, text)

	print("eMail sent")
