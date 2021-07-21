import cv2 
import numpy as py
import os
import time
from gpiozero import LED

ledBase = LED(27)
ledTop = LED(17)
ledBase.off()
ledTop.off()
time.sleep(.1)
ledTop.on()
ledBase.on()
time.sleep(.1)

if __name__ == "__main__":
	ledTop.off()
	#print("Top On before photo")
	time.sleep(1)
	os.system('raspistill -t 500 -o /home/pi/Documents/unfish/orig/TopView.jpg')
	ledTop.on()
	print("Photo 1 taken")
	time.sleep(.1)
	ledBase.off()
	#print("base on taking photo")
	time.sleep(1)
	os.system('raspistill -t 500 -o /home/pi/Documents/unfish/orig/workingImage.jpg')
	ledBase.on()
	print("Photo 2 taken")
	time.sleep(.1)
