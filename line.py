

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from skimage import measure
import overlap
import take_reference as ref
import math
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import csv
from mpldatacursor import datacursor

#specify the area to crop the square in the reference plane
'''height =1200
width=1500
template_x=600
template_y=300'''
height =2200
width=2800
template_x=0
template_y=0
line=[]
dat=[]
line1=[]
slope_int=[]



##########find mid point of given two points########
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


#########draw graph boxes########
def draw_lines(image):
       for k in range( 0,width,10):
        cv2.line(image,(k,0),(k,height),(255,255,255),1)
       for k in range (0,height,10): 
        cv2.line(image,  (0,k),(width,k),(255,255,255),1)








####read the dimension to pixel ratio from a csv file
def get_ratio():        
 with open('ratio.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        return row[0]
        
    
        
### calculate distance between two parallel lines with new method
def cal_new(slope_int,number_of_lines):
  
  if number_of_lines==2:
   numerator= abs(slope_int[1][1]-slope_int[0][1])
   denominator= math.sqrt(1+(slope_int[0][0])*(slope_int[0][0]))
   distance_pixel= numerator/denominator
   print distance_pixel
   distance= distance_pixel*float(get_ratio())
   print "distance",distance,"inch"

  elif number_of_lines==4:
   numerator= abs((slope_int[1][1]+slope_int[0][1])/2-(slope_int[2][1]+slope_int[3][1])/2)
   denominator= math.sqrt(1+((slope_int[0][0]+slope_int[1][0])/2)*((slope_int[2][0]+slope_int[3][0])/2))
   distance_pixel= numerator/denominator
   print distance_pixel
   distance= distance_pixel*float(get_ratio())
   print "distance",distance,"inch"

  else:
   print "image not clear"
          
          



#calculate slope and distance between two lines using old method
def cal_dst(line,number_of_lines):
 if number_of_lines==2:       
  m=(line[1][1]-line[0][1])/(line[1][0]-line[0][0]) #slope of the line
  #m= (line[3][1]-line[2][1])/(line[3][0]-line[2][0])
  numerator= abs(m*((line[2][0])-(line[0][0]))+((line[0][1])-(line[2][1])))
  print numerator
  denominator= math.sqrt(1+m*m)
  print denominator
  distance_pixel= numerator/denominator
  print distance_pixel
  distance= distance_pixel*ref.find_ref("ref2.png")
  print "distance",distance,"inch"

 elif number_of_lines==4:
  slope_num=(((line[1][1]+line[3][1])/2)-((line[0][1]+line[2][1])/2))
  slope_den= (((line[1][0]+line[3][0])/2)-((line[0][0]+line[2][0])/2))
  m= slope_num/slope_den
  #numerator= abs(m*((line[2][0])-(line[0][0]))+((line[0][1])-(line[2][1])))
  #m= (line[3][1]-line[2][1])/(line[3][0]-line[2][0])
  print m
  numerator= abs(m*((line[4][0]+line[6][0])/2)-((line[0][0]+line[2][0])/2))+(((line[0][1]+line[2][1])/2)-((line[4][1]+line[6][1])/2))
  print numerator
  denominator= math.sqrt(1+m*m)
  print denominator
  distance_pixel= numerator/denominator
  print distance_pixel
  distance= distance_pixel*ref.find_ref("ref2.png")
  print "distance",distance,"inch"

 else:
  print "image not clear"
        
 
        
def find_bend(img):


 #for cropping the 
 '''height =800
 width=600
 template_x=200
 template_y=10'''
 height =2200
 width=2800
 template_x=0
 template_y=0
 
 # load the image, crop it and  convert it to grayscale
 image= cv2.imread(img)
 if image is None:
         print "No image"
         return 1
 image= image[template_y:template_y+height,template_x:template_x+width]
 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 


 # perform edge detection, then perform a dilation + erosion to
 # close gaps in between object edges
 edged = cv2.Canny(gray,50,100)
 edged1 = cv2.dilate(edged, None, iterations=2)
 edged2 = cv2.erode(edged1, None, iterations=1)
 
 

 
 # find contours in the edge map
 cnts = cv2.findContours(edged2.copy(),  cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
 cnts = cnts[0] if imutils.is_cv2() else cnts[1]
 count_line=0
 for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 2000:
		continue
        cv2.drawContours(image, c, -1, (254, 0, 0), 1)
        #print len(c)
        dat.append(c)
        count_line=count_line+1


 #print count_line
 #print len(dat)
 #print dat
 regression (dat,count_line)
 return 1
 cv2.namedWindow('image',cv2.WINDOW_NORMAL)
 cv2.resizeWindow('image', 3200,2200)
 cv2.imshow("image",image)
 #cv2.namedWindow('Erode',cv2.WINDOW_NORMAL)
 #cv2.resizeWindow('Erode', 1000,600)
 cv2.imshow("Erode", edged2)
 cv2.waitKey(0)

 
def regression(dat,count_line):

      

   for k in range(len(dat)): 
     #print k
     data = [ a[0] for a in dat[k]]
     regr = linear_model.LinearRegression() #for horizontal lines
     regr1 = linear_model.LinearRegression() #for vertical lines
     ppx, ppy = map(list, zip(*data))
     x_= np.array(ppx)
     y_=np.array(ppy)
     x_ = x_[:, np.newaxis]
     y_=y_[:,np.newaxis]
     
     # Train the model using the training sets
     regr.fit(x_,y_)# horizontal
     regr1.fit(y_,x_)# vertical
     
     #print x
     #print y
     bending_y_pred = regr.predict(x_)
     bending_x_pred=regr1.predict(y_)
     #print bending_y_pred
     #print len(bending_y_pred)
     # The coefficients
     print 'Coefficients[horizontal]: ', regr.coef_,regr.intercept_
     print 'Coefficients[vertical]: ',regr1.coef_,regr1.intercept_

     
     
     err=mean_squared_error(y_, bending_y_pred)
     err1=mean_squared_error(x_,bending_x_pred)


     if err<= err1:
      max_x = max(x_)[0]
      min_x = min(x_)[0]
      max_y=regr.predict(max_x)[0][0]
      #max_y=max_y[0]
      min_y=regr.predict(min_x)[0][0]
      #min_y=min_y[0]
      line1.append((min_x,min_y))
      line1.append((max_x,max_y))
      slope_int.append((regr.coef_[0][0],regr.intercept_[0]))
      #print line1 
     
      max_error=0 #supposed values 
      min_error= 100
      for i in range(len(y_)):
             error= abs(y_[i]-bending_y_pred[i])
             #print error
             if error > max_error:
                     max_error=error
                     place= x_[i]
                     
             if error< min_error:
                     min_error= error
                     
      print "max error",max_error*float(get_ratio()), "in the pixel",place*float(get_ratio())
      print "min error",min_error*float(get_ratio())

     else:
        max_y=max(y_)[0]
        min_y=min(y_)[0]
        max_x= regr1.predict(max_y)[0][0]
        min_x= regr1.predict(min_y)[0][0]
        line1.append((min_y,min_x))
        line1.append((max_y,max_x))
        slope_int.append((regr1.coef_[0][0],regr1.intercept_[0]))
        #print line1 
     
        max_error=0 #supposed values 
        min_error= 100
        for i in range(len(x_)):
             error= abs(x_[i]-bending_x_pred[i])
             #print error
             if error > max_error:
                     max_error=error
                     place= y_[i]
                     
             if error< min_error:
                     min_error= error
                     
        print "max error",max_error*float(get_ratio()), "in the pixel",place*float(get_ratio())
        print "min error",min_error*float(get_ratio())
        
     # The mean squared error
     print "Mean squared error [horizontal],[vertical]: ",err ,err1
     print "RMSE [horizontal],[vertical]:  ",math.sqrt(err), math.sqrt(err1)
     plt.scatter(x_*float(get_ratio()),y_*float(get_ratio()),  color='green')
     plt.plot(x_*float(get_ratio()), bending_y_pred*float(get_ratio()), 'r')
     plt.plot(bending_x_pred*float(get_ratio()),y_*float(get_ratio()),'b')
     plt.xlim(0,2200*float(get_ratio()))
     plt.ylim(0,3200*float(get_ratio()))
   #print slope_int  
   #print line1
   
   datacursor( display='multiple',arrowprops=dict(arrowstyle='simple', fc='green', alpha=0.5),draggable=True,
               formatter="X_axis:{x:.2f} inch\nY_axis:{y:.2f} inch".format)
   cal_new(slope_int,count_line)
   plt.show()
        

        
def find_line(img):
        
 # load the image, crop it and  convert it to grayscale
 image= cv2.imread(img)
 image= image[template_y:template_y+height,template_x:template_x+width]
 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 ret,th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
 #gray = cv2.bilateralFilter(gray,9,75,75)
 #gray = cv2.GaussianBlur(gray, (7, 7), 0)


 # perform edge detection, then perform a dilation + erosion to
 # close gaps in between object edges
 edged = cv2.Canny(gray,50,100)
 edged1 = cv2.dilate(edged, None, iterations=2)
 edged2 = cv2.erode(edged1, None, iterations=1)
 #minLineLength = 1500
 #maxLineGap = 20
 image2=edged1.copy()
 cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

 
 # find contours in the edge map
 cnts = cv2.findContours(edged2.copy(),  cv2.RETR_TREE,
	cv2.CHAIN_APPROX_SIMPLE)
 cnts = cnts[0] if imutils.is_cv2() else cnts[1]

 

 
        
 #draw_lines(image) # draw the graph lines

 number_of_lines=0
 for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 1000:
		continue
    
	# compute the rotated bounding box of the contour
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding box
	box = perspective.order_points(box)
        cv2.drawContours(image,  [box.astype("int")], -1, (255, 0, 0), 2)
        number_of_lines=number_of_lines+1
        #print box.astype("int")
        mid1= midpoint(box[0],box[3])
        mid2= midpoint(box[1],box[2])
        cv2.line(image,(int(mid1[0]),int(mid1[1])), (int(mid2[0]),int(mid2[1])),(0, 243, 0), 2)
        line.append(mid1)
        line.append(mid2)
        
        #cv2.drawContours(image, cnts, -1, (0, 0, 0), 2)

 
 #print line
 #print number_of_lines
 cal_dst(line,number_of_lines)
 
         
 
         

 
 
 cv2.namedWindow('image',cv2.WINDOW_NORMAL)
 cv2.resizeWindow('image', 1000,600)
 cv2.imshow("image",image)
 cv2.namedWindow('Gray',cv2.WINDOW_NORMAL)
 cv2.resizeWindow('Gray', 1000,600)
 cv2.imshow("Gray", gray)
 cv2.namedWindow('Binary_Threshold',cv2.WINDOW_NORMAL)
 cv2.resizeWindow('Binary_Threshold', 1000,600)
 cv2.imshow("Binary_Threshold", th1)
 cv2.namedWindow('Edged',cv2.WINDOW_NORMAL)
 cv2.resizeWindow('Edged', 1000,600)
 cv2.imshow("Edged", edged)
 cv2.namedWindow('dilate',cv2.WINDOW_NORMAL)
 cv2.resizeWindow('dilate', 1000,600)
 cv2.imshow("dilate", edged1)
 cv2.namedWindow('Erode',cv2.WINDOW_NORMAL)
 cv2.resizeWindow('Erode', 1000,600)
 cv2.imshow("Erode", edged2)
 cv2.imshow("Overplap", image2)
 cv2.waitKey(0)


