source virtualenvwrapper.sh
workon py3cv4
if [ $1 != '-h' ] 
then
#Includes UNFISH
   cd /home/pi/Documents/unfish
   filename='/home/pi/Documents/unfish/corrected_images/workingImage.jpg'
   touch $filename
   if [ -f $filename ]; then
      rm /home/pi/Documents/unfish/corrected_images/workingImage.jpg
   fi
   filename='/home/pi/Documents/unfish/corrected_images/TopView.jpg'
   touch $filename
   if [ -f $filename ]; then
      rm /home/pi/Documents/unfish/corrected_images/TopView.jpg
   fi
#Includes no DIMS
#   echo ------------------------
#   echo Enter the X dimension:
#   read x_dim
#   echo Enter the Y dimension:
#   read y_dim
#   echo Enter the Z dimension:
#   read z_dim
   x_dim="0.00"
   y_dim="0.00"
   z_dim="0.00"
   echo Enter the tool number or drawer description **must be one word _ ok**:
   read toolnumber
   echo ------------------------
#Includes RPI camera
   python /home/pi/Documents/betterprocess/fullsize/rpicamerafullsize.py
   unfish apply orig/*
fi
python /home/pi/Documents/betterprocess/fullsize/outlinefullsize.py $1 $z_dim $toolnumber $x_dim $y_dim
if [ $1 != '-h' ] 
then
   sleep 8
   pkill -15 gpicview
fi 
