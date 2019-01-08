from time import sleep
from picamera import PiCamera



###########this module captures the image and saves in .png format#######

def take_image( image):
 try:
     res= (3280,2464)
     camera = PiCamera()
     camera.resolution = res
     #(2592,1944) #other camera resolution
     camera.iso = 100
     camera.start_preview()
 
     # Camera warm-up time
     sleep(2)
 
     '''####change the camera parameters if required #######
     camera.shutter_speed = camera.exposure_speed
     camera.exposure_mode = 'off'
     g = camera.awb_gains
     camera.awb_mode = 'off'
     camera.awb_gains = g'''
 
     camera.capture(image)
     print "Photo taken at", res, "resolution"

 finally:
      camera.close()
