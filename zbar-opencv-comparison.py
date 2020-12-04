import numpy as np
import sys
import time
import pyzbar.pyzbar as pyzbar
import math
import cv2
import copy
import pandas as pd
import time
from main import *

#FX = 7.2125114092664523e+02
FX = 7.304041689e+02
SIDE_OF_QR = 45
# ANGLE_FI=24*math.pi/180
# ANGLE_MU=19*math.pi/180
ANGLE_FI=45*math.pi/180
ANGLE_MU=45*math.pi/180
H_QR = -20
H_CAMERA = 110

VIDEO_NAME="video1.avi"
TEST_NAME="test1"


df=pd.DataFrame(columns=['t','x','y','alpha','f'])


class Point:
  def __init__(self,x,y):
    self.x=x
    self.y=y

cap = cv2.VideoCapture(0)
hasFrame,frame = cap.read()
out = cv2.VideoWriter(VIDEO_NAME,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))
CX=frame.shape[1]/2
CY=frame.shape[0]/2



# Display barcode and QR code location
def display(im, decodedObjects):

  # Loop over all decoded objects
  for decodedObject in decodedObjects:
    points = decodedObject.polygon

    # If the points do not form a quad, find convex hull
    if len(points) > 4 :
      hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
      hull = list(map(tuple, np.squeeze(hull)))
    else :
      hull = points;

    # Number of points in the convex hull
    n = len(hull)

    # Draw the convext hull
    for j in range(0,n):
      cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)


def distanceCalculate(point1, point2):
  Len=((point1.x-point2.x)**2+(point1.y-point2.y)**2)**0.5       
  Z=FX*SIDE_OF_QR/(Len)
  return Z

def getCenter(point1, point2):
  p = Point(0,0)
  p.x=(point1.x)+((point2.x)-(point1.x))/2
  p.y=(point1.y)+((point2.y-point1.y))/2
  return p

def distanceCalculate2(point1,point2,H):
  Z=distanceCalculate(point1,point2)#/math.cos(angle)
  L = (Z**2 - H**2)**0.5
  return L

def getMU(Y):
  return math.atan2(Y-CY,CY)

def getF(X):
  return math.atan2(X-CX,CX)


# Detect and decode the qrcode
Data=None
t = time.time()
time_before=t

index=0
start_time = time.time()
while(1):
    hasFrame, inputImage = cap.read()
    if not hasFrame:
        break
    decodedObjects = pyzbar.decode(inputImage)

    if len(decodedObjects):
        zbarData = decodedObjects[0].data
    else:
        zbarData=''
    if zbarData:
        arr = list(map(float, zbarData.split()))
        SIDE_OF_QR = arr[0]
        H_QR = arr[3] - H_CAMERA
        cv2.putText(inputImage, "ZBAR : {}".format(zbarData), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        polygon = decodedObjects[0].polygon


        data=polygon[:]
        if ((polygon[0].y + 30)<(polygon[1].y)):
          data=polygon[:]
          key=False
        else:
          key=True
          data[0]=polygon[3]
          data[1]=polygon[0]
          data[2]=polygon[1]
          data[3]=polygon[2]
          #cv2.putText(inputImage, f"Rotated !!!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)            
          

        #cv2.circle(inputImage,data[0],20,(0,255,0),5)# зеленый - левый вверх
        #cv2.circle(inputImage,data[1],20,(255,0,0),5) # синий - левый низ
        #cv2.circle(inputImage,data[2],20,(0,0,255),5) # red - правый низ
        #cv2.circle(inputImage,data[3],20,(0,0,0),5)# black - правый верх
        
        #mu_0= getMU(getCenter(data[0], data[3]).y + (getCenter(data[1], data[2]).y - getCenter(data[0], data[3]).y)/2)
        
        f_0= getMU(getCenter(data[0], data[3]).x + (getCenter(data[1], data[2]).x - getCenter(data[0], data[3]).x)/2)

        a = distanceCalculate2(data[0], data[1], H_QR)
        b = distanceCalculate2(getCenter(data[0], data[3]), getCenter(data[1], data[2]), H_QR)
        c = SIDE_OF_QR/2
        cosA = (a**2 - b**2 - c**2)/(-2*b*c)
        Arg = np.arccos(cosA)#*180/math.pi
        x = b * math.sin(Arg)
        y = b * math.cos(Arg)

        cv2.putText(inputImage, f"Distance = {round(b,3)}, Alpha = {round(Arg,3)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA) 

        current_time=time.time()
        elapsed_time_secs = current_time - time_before
        time_before=current_time

        x_p,p_p=Prediction(dt,x_c,p_c)
        Y=np.asarray([x,y,Arg])
        x_c,p_c=Correction(Y,elapsed_time_secs,x_p,p_p)

        cv2.putText(inputImage, f"X = {round(x_c[0], 3)}, Y = {round(x_c[1],3)} ", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

        df.loc[index]={'t':elapsed_time_secs,'x':x,'y':y,'alpha':f_0,'f':Arg}
        index=index+1; df.to_csv(TEST_NAME)

    else:
        cv2.putText(inputImage, "ZBAR : QR Code NOT Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    display(inputImage, decodedObjects)
    #cv2.imshow("Result",inputImage)

    out.write(inputImage)
    k = cv2.waitKey(20)
    if k == 27:
        break
cv2.destroyAllWindows()
#vid_writer.release()
