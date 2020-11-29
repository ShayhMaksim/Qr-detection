import pandas as pd
import numpy as np
from numpy import linalg as LA

def Prediction(dt:float,x_c,p_c):  
    F=np.asarray([
        [1,0],
        [0,1],
    ])
    x_p=np.dot(F,x_c)
    p_p=np.dot(np.dot(F,p_c),np.transpose(F))
    return x_p,p_p

def Correction(Y,dt:float,x_p,p_p):
    Dphi_1=LA.inv(Dphi)
    
    p_c=np.copy(LA.inv(np.add(LA.inv(p_p),np.dot(np.dot(H,Dphi_1),H))))
    
    mult=np.dot(np.dot(p_c,H),Dphi_1)
    
    subl=Y-np.dot(H,x_p)
    
    x_c=np.copy(np.add(x_p,np.dot(mult,subl)))

    return x_c,p_c

    
df = pd.read_csv('Tests4')

t=0.
dt=0.001
index=0
current_t=0.

New_X=pd.DataFrame(columns=['t','x','y'])

New_P=pd.DataFrame(columns=['t','p_p_X','p_p_Y'])

current_t=df['t'][index]

x_p=np.zeros(2)
x_c=np.zeros(2)

#ошибки измерений (экспертное мнение)
Dphi=np.zeros((2,2))
Dphi[0][0]=2
Dphi[1][1]=2


#матрица наблюдений
H=np.zeros((2,2))
H[0][0]=1
H[1][1]=1

p_p=np.zeros((2,2))
p_c=np.zeros((2,2))

p_c[0][0]=10
p_c[1][1]=10

x_c[0]=800
x_c[1]=200


special_t=0

while(1):

    if ((t+dt)>current_t):
        x_p,p_p=Prediction(df['t'][index],x_c,p_c)
        Y=np.asarray([df['x'][index],df['y'][index]])
        x_c,p_c=Correction(Y,df['t'][index],x_p,p_p)

        New_X.loc[index]={'t':t,'x':x_c[0],'y':x_c[1],}
        New_P.loc[index]={'t':t,'p_p_X':p_c[0][0],'p_p_Y':p_c[1][1],}
        index=index+1
        current_t=df['t'][index]+current_t
        special_t=0
        if (index==9): break

    t=t+dt
    special_t=special_t+dt
    
New_X.to_csv('X2')
New_P.to_csv('P2')