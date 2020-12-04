import pandas as pd
import numpy as np
from numpy import linalg as LA

def Prediction(dt:float,x_c,p_c):  
    F=np.asarray([
        [1.,0.,-V*dt*np.cos(x_c[2])],
        [0.,1.,-V*dt*np.sin(x_c[2])],
        [0.,0.,1.]
    ])
    x_p=np.zeros(3)
    x_p[0]=x_c[0]-V*dt*np.sin(x_c[2])
    x_p[1]=x_c[1]+V*dt*np.cos(x_c[2])
    x_p[2]=x_c[2]

    p_p=np.dot(np.dot(F,p_c),np.transpose(F))
    return x_p,p_p

def Correction(Y,dt:float,x_p,p_p):
    Dphi_1=LA.inv(Dphi)
    
    p_c=np.copy(LA.inv(np.add(LA.inv(p_p),np.dot(np.dot(H,Dphi_1),H))))
    
    mult=np.dot(np.dot(p_c,H),Dphi_1)
    
    subl=Y-np.dot(H,x_p)
    
    x_c=np.copy(np.add(x_p,np.dot(mult,subl)))

    return x_c,p_c

    
df = pd.read_csv('Tests13')

print(df.head())

t=0.
dt=0.0001
index=0
current_t=0.

New_X=pd.DataFrame(columns=['t','x','y','alpha'])

New_P=pd.DataFrame(columns=['t','p_p_X','p_p_Y','p_p_Aplha'])

current_t=df['t'][index]



#ошибки измерений (экспертное мнение)
Dphi=np.zeros((3,3))
Dphi[0][0]=9
Dphi[1][1]=9
Dphi[2][2]=0.02

#матрица наблюдений
H=np.zeros((3,3))
H[0][0]=1
H[1][1]=1
H[2][2]=1

p_p=np.zeros((3,3))
p_c=np.zeros((3,3))

p_c[0][0]=50
p_c[1][1]=50
p_c[2][2]=50

V=130

x_p=np.zeros(3)
x_c=np.zeros(3)

x_c[0]=1300
x_c[1]=0
x_c[2]=1.56

special_t=0

# x=np.zeros(3)
# x[:]=x_c[:]


# while(1):
  
#     x[0]=x[0]-V*dt*np.sin(x[2])
#     x[1]=x[1]+V*dt*np.cos(x[2])
#     x[2]=x[2]

#     # x_p,p_p=Prediction(dt,x,p_c)
#     # x[:]=x_p[:]
#     # p_c[:]=p_c[:]

#     if ((t+dt)>current_t):
#         x_p,p_p=Prediction(dt,x,p_c)
#         Y=np.asarray([df['x'][index],df['y'][index],df['f'][index]])
#         x_c,p_c=Correction(Y,df['t'][index],x_p,p_p)
#         x[:]=x_c[:]
#         New_X.loc[index]={'t':current_t,'x':x_c[0],'y':x_c[1],'alpha':x_c[2],}
#         New_P.loc[index]={'t':current_t,'p_p_X':p_c[0][0],'p_p_Y':p_c[1][1],'p_p_Aplha':p_c[2][2],}
#         index=index+1
#         if (index==12): break
#         current_t=df['t'][index]+current_t
#         special_t=0

#     t=t+dt
#     special_t=special_t+dt
    
# New_X.to_csv('X')
# New_P.to_csv('P')