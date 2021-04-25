from ctypes import string_at
import numpy as np
import sys
import time
from numpy.core.fromnumeric import mean
from numpy.core.overrides import ArgSpec
from pandas.core.indexes import base
import pyzbar.pyzbar as pyzbar
import math
import cv2
import copy
import pandas as pd
import time
from main2 import *
import socket
import threading
import scipy.optimize 

#FX = 7.2125114092664523e+02
FX = 7.304041689e+02
#SIDE_OF_QR = 45
# ANGLE_FI=24*math.pi/180
# ANGLE_MU=19*math.pi/180
ANGLE_FI=45*math.pi/180
ANGLE_MU=45*math.pi/180
H_QR = -20
H_CAMERA = 45#1900#45#110#1900


VIDEO_NAME="F4.avi"
TEST_NAME="test444"
REAL_DATA="Exp24"
SUPPORT_DATA="Exp24_s"

# сетевое программирование для межпрограммного взаимодействие
# BIND_IP='127.0.0.1'
# BIND_PORT=8080
# server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# server.bind((BIND_IP,BIND_PORT))
# server.listen(2) # max blocklog of connections
# print('Listening on {}:{}'.format(BIND_IP,BIND_PORT))
# client_sock,address=server.accept()
# print('Accepted connection from {}:{}'.format(address[0],address[1]))

# handle для отправки сообщения клиенту
# def handle_client_connection(client_socket,message):
#   #request=client_socket.recv(1024)
#   #print('Received {}'.format(request))
#   client_socket.send(message.encode())


supportDF=pd.DataFrame({'t':[],'x':[],'y':[]})
globalDF=pd.DataFrame({'t':[],'x':[],'y':[]})

class Point:
  def __init__(self,x,y):
    self.x=x
    self.y=y

cap = cv2.VideoCapture(0)
hasFrame,frame = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter(VIDEO_NAME,fourcc, 10, (frame.shape[1],frame.shape[0]),0)
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


# нахождене расстояния до Qr-кода
def distanceCalculate(point1, point2,SIDE_OF_QR):
  Len=((point1.x-point2.x)**2+(point1.y-point2.y)**2)**0.5       
  Z=FX*SIDE_OF_QR/(Len)
  return Z

# нахождене расстояния до Qr-кода
def coordY(point1, point2, X,SIDE_OF_QR):
  Len=((point1.x-point2.x)**2+(point1.y-point2.y)**2)**0.5       
  y=SIDE_OF_QR*(X-CX)/Len
  return y

# нахождение центральной точки
def getCenter(point1, point2):
  p = Point(0,0)
  p.x=(point1.x)+((point2.x)-(point1.x))/2
  p.y=(point1.y)+((point2.y-point1.y))/2
  return p

def distanceCalculate2(point1,point2,H,SIDE_OF_QR):
  Z=distanceCalculate(point1,point2,SIDE_OF_QR)#/math.cos(angle)
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


# Данные в Qt коде - [size z x y angle]

def SMA(y,count):
  try:
    if SMA.count<=count:
      SMA.count+=1
      SMA.data.append(y)
    else:
      del SMA.data[0]
      SMA.data.append(y)
    return np.sum(SMA.data)/SMA.count
  except:
    SMA.count=1
    SMA.data=[]
    SMA.data.append(y)
    return y

def funDeltaX(D,phi):
    return 26.98145955-0.08641856*D/np.sin(phi)

def funDeltaY(D,phi):
    return -0.20324165*D*np.cos(phi)


def SingleData(inputImage,decodedObjects,textStep):
    zbarData = decodedObjects.data
    arr = list(map(float, zbarData.split()))
    SIDE_OF_QR = arr[0]
    H_QR = arr[1] - H_CAMERA
    cv2.putText(inputImage, "ZBAR : {}".format(zbarData), (10, 50+textStep), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    polygon = decodedObjects.polygon


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
      # cv2.putText(inputImage, f"Rotated !!!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)            
          

      # cv2.circle(inputImage,data[0],20,(0,255,0),5)# зеленый - левый вверх
      # cv2.circle(inputImage,data[1],20,(255,0,0),5) # синий - левый низ
      # cv2.circle(inputImage,data[2],20,(0,0,255),5) # red - правый низ
      # cv2.circle(inputImage,data[3],20,(0,0,0),5)# black - правый верх
        
        
    centerTop=getCenter(data[0], data[3])
    centerBottom=getCenter(data[1], data[2])


    a = distanceCalculate2(data[0], data[1], H_QR,SIDE_OF_QR)
    b = distanceCalculate2(centerTop, centerBottom, H_QR,SIDE_OF_QR)
    c = SIDE_OF_QR/2
    d = distanceCalculate2(data[2], data[3], H_QR,SIDE_OF_QR)
    cosA = (a**2 - b**2 - c**2)/(-2*b*c)
    cosB = (d**2 - b**2 - c**2)/(-2*b*c)
    A = np.arccos(cosA)
    B = np.arccos(cosB)

    Arg=B+(np.pi-A-B)/2

    X=(centerTop.x+centerBottom.x)/2.
        
    b = mean([a,b,d])
    b=b/math.sin(Arg)
    x = b *math.sin(Arg) 


    k=1./math.sin(Arg)


    dY =  coordY(centerTop, centerBottom,(centerTop.x+centerBottom.x)/2.,SIDE_OF_QR)*k
    y = b *math.cos(Arg) - dY
    y=SMA(y,2)
    Arg=math.atan2(x,y)
    b = (x**2+y**2)**0.5
    dx=funDeltaX(b,Arg)
    x = x + dx

    b = (x**2+y**2)**0.5
    Arg=math.atan2(x,y)

    cv2.putText(inputImage, f"Distance = {round(b,3)}, Alpha = {round(Arg,3)}", (10, 70+textStep), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)     
    cv2.putText(inputImage, f"X = {round(x, 3)}, Y = {round(y,3)} ", (10, 90+textStep), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

    globalX=x*np.cos(np.pi/180*arr[4])-y*np.sin(np.pi/180*arr[4])+arr[2]
    globalY=x*np.sin(np.pi/180*arr[4])+y*np.cos(np.pi/180*arr[4])+arr[3]
    cv2.putText(inputImage, f"X(gl) = {round(globalX, 3)}, Y(gl) = {round(globalY,3)} ", (10, 110+textStep), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

    return globalX,globalY


def Localization(x,b):
    x, y = x
    X=[]
    for i in b:
        X.append([i[0],i[1]])
    Result=[]
    for x1 in X:
        Result.append((x1[0]-x)**2+(x1[1]-y)**2)
    return np.asarray(Result)

def RP(x,d,b):
    return (Localization(x,b)-d)


def SingleData2(inputImage,decodedObjects,textStep):
    zbarData = decodedObjects.data
    arr = list(map(float, zbarData.split()))
    SIDE_OF_QR = arr[0]
    H_QR = arr[1] - H_CAMERA
    cv2.putText(inputImage, "ZBAR : {}".format(zbarData), (10, 50+textStep), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    polygon = decodedObjects.polygon


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

        
    centerTop=getCenter(data[0], data[3])
    centerBottom=getCenter(data[1], data[2])


    a = distanceCalculate2(data[0], data[1], H_QR,SIDE_OF_QR)
    b = distanceCalculate2(centerTop, centerBottom, H_QR,SIDE_OF_QR)
    c = SIDE_OF_QR/2
    d = distanceCalculate2(data[2], data[3], H_QR,SIDE_OF_QR)
    cosA = (a**2 - b**2 - c**2)/(-2*b*c)
    cosB = (d**2 - b**2 - c**2)/(-2*b*c)
    A = np.arccos(cosA)
    B = np.arccos(cosB)

    Arg=B+(np.pi-A-B)/2

        
    b = mean([a,b,d])
    b=b/math.sin(Arg)
    x = b *math.sin(Arg)
    dY =  coordY(centerTop, centerBottom,(centerTop.x+centerBottom.x)/2.,SIDE_OF_QR)/(math.sin(Arg))  #math.cos(Arg)
    y = b *math.cos(Arg) - dY
    y=SMA(y,2)
    b = (x**2+y**2)**0.5
    Arg=math.atan2(x,y)

    Data=[]

    dX=0*np.cos(np.pi/180*arr[4])-(0-c)*np.sin(np.pi/180*arr[4])+arr[2]
    dY=0*np.sin(np.pi/180*arr[4])+(0-c)*np.cos(np.pi/180*arr[4])+arr[3]
    Data.append([arr[2]+dX,arr[3]+dY])

    dX=0*np.cos(np.pi/180*arr[4])-(0+c)*np.sin(np.pi/180*arr[4])+arr[2]
    dY=0*np.sin(np.pi/180*arr[4])+(0+c)*np.cos(np.pi/180*arr[4])+arr[3]
    Data.append([arr[2]+dX,arr[3]+dY])
    
    X0=np.asarray([x,y])
    Distance=[a*a,d*d]
    res = scipy.optimize.leastsq(RP, X0, args=(Distance,Data))
    x_ = res[0]


    cv2.putText(inputImage, f"Distance = {round(b,3)}, Alpha = {round(Arg,3)}", (10, 70+textStep), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)     
    cv2.putText(inputImage, f"X = {round(x, 3)}, Y = {round(y,3)} ", (10, 90+textStep), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

    # globalX=x*np.cos(np.pi/180*arr[4])-y*np.sin(np.pi/180*arr[4])+arr[2]
    # globalY=x*np.sin(np.pi/180*arr[4])+y*np.cos(np.pi/180*arr[4])+arr[3]
    cv2.putText(inputImage, f"X(gl) = {round(x_[0], 3)}, Y(gl) = {round(x_[1],3)} ", (10, 110+textStep), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

    return x_[0],x_[1]

def Classic(inputImage,decodedObjects,X0):
  Data=[]
  Distance=[]

  for i in range(0,len(decodedObjects)):
    zbarData = decodedObjects[i].data
    arr = list(map(float, zbarData.split()))
    Data.append([arr[2],arr[3]])
    polygon = decodedObjects[i].polygon
    SIDE_OF_QR = arr[0]
    H_QR = arr[1] - H_CAMERA

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
       
    centerTop=getCenter(data[0], data[3])
    centerBottom=getCenter(data[1], data[2])


    a = distanceCalculate2(data[0], data[1], H_QR,SIDE_OF_QR)
    b = distanceCalculate2(centerTop, centerBottom, H_QR,SIDE_OF_QR)
    d = distanceCalculate2(data[2], data[3], H_QR,SIDE_OF_QR)
    # c = SIDE_OF_QR/2

    # cosA = (a**2 - b**2 - c**2)/(-2*b*c)
    # cosB = (d**2 - b**2 - c**2)/(-2*b*c)
    # A = np.arccos(cosA)
    # B = np.arccos(cosB)

    # Arg=B+(np.pi-A-B)/2

    b = mean([a,b,d])
    
    # b=b/math.sin(Arg)
    # x = b *math.sin(Arg)
    # dY =  coordY(centerTop, centerBottom,(centerTop.x+centerBottom.x)/2.,SIDE_OF_QR)/(math.sin(Arg))  #math.cos(Arg)
    # y = b *math.cos(Arg) - dY
    # y=SMA(y,2)
    # b = (x**2+y**2)**0.5

    # X=(centerTop.x+centerBottom.x)/2.
    # dY=  arr[3]*(X-CX)/CX
    b = (b*b)

    Distance.append(b)
  res = scipy.optimize.leastsq(RP, X0, args=(Distance,Data))
  x = res[0]

  cv2.putText(inputImage, f"X(gl) = {round(x[0], 3)}, Y(gl) = {round(x[1],3)} ", (320, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

  return x[0],x[1]
    

while(1):
    hasFrame, inputImage = cap.read()
    inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    

    if not hasFrame:
        break
    decodedObjects = pyzbar.decode(inputImage)

    
    if len(decodedObjects):
      x=0
      y=0
      for i in range(0,len(decodedObjects)):
        x1,y1=SingleData(inputImage,decodedObjects[i],i*110)
        x+=x1
        y+=y1
      x=x/len(decodedObjects)
      y=y/len(decodedObjects)

      current_time=time.time()
      elapsed_time_secs = current_time - time_before
      time_before=current_time

      cv2.putText(inputImage, f"Single X = {round(x, 3)}, Y = {round(y,3)} ", (320, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
       

      if len(decodedObjects)>1:
        X0=np.asarray([x,y])
        x_2,y_2=Classic(inputImage,decodedObjects,X0)
        supportDF=supportDF.append({'t':index,'x':x_2,'y':y_2},ignore_index=True)
        

      
      globalDF=globalDF.append({'t':index,'x':x,'y':y},ignore_index=True)


      index=index+1; 

      globalDF.to_csv(REAL_DATA)
      supportDF.to_csv(SUPPORT_DATA)

        #message=str(elapsed_time_secs)+','+str(globalX)+','+str(globalY)
        #если не разделять по потокам, то будет смешение данных
        # client_handler=threading.Thread(
        #   target=handle_client_connection,
        #   args=(client_sock,message,)
        # )
        # client_handler.start()

    else:
        cv2.putText(inputImage, "ZBAR : QR Code NOT Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    #inputImage = cv2.cvtColor(inputImage, cv2.COLOR_GRAY2BGR)
    display(inputImage, decodedObjects)
    cv2.imshow("Result",inputImage)

    out.write(inputImage)
    k = cv2.waitKey(20)
    if k == 27:
        break
cv2.destroyAllWindows()
#client_sock.close()
#vid_writer.release()





