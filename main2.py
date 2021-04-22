import pandas as pd
import numpy as np
from numpy import linalg as LA

E=np.asarray([
        [4,0,0,0],
        [0,4,0,0],
        [0,0,0,0],
        [0,0,0,0]
    ])

def Prediction(dt:float,x_c,p_c):  
    F=np.asarray([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])
    x_p=np.dot(F,x_c)
    
    p_p=np.add(E,np.dot(np.dot(F,p_c),np.transpose(F)))
    
    return x_p,p_p

def Correction(Y,dt:float,x_p,p_p,D,angle):
    H=np.asarray([
        [1,0,0,0],
        # [
        #     -x_p[0]*x_p[1]*x_p[3]/((x_p[1]**2+x_p[0]**2)**1.5),
        #     1-x_p[1]*x_p[1]*x_p[3]/((x_p[1]**2+x_p[0]**2)**1.5)+x_p[3]/((x_p[1]**2+x_p[0]**2)**0.5),
        #     0,
        #     x_p[1]/((x_p[1]**2+x_p[0]**2)**0.5)
        # ],
        [0,1,0,0]
    ])
    H_=np.transpose(H)
    Dphi_1=LA.inv(Dphi)
    
    p_c=np.copy(LA.inv(np.add(LA.inv(p_p),np.dot(np.dot(H_,Dphi_1),H))))
    
    mult=np.dot(np.dot(p_c,H_),Dphi_1)
    
    subl=Y-np.dot(H,x_p)
    
    x_c=np.copy(np.add(x_p,np.dot(mult,subl)))

    return x_c,p_c


# df = pd.read_csv('Tests4')


t=0.
dt=0.001
index=0
current_t=0.

# New_X=pd.DataFrame(columns=['t','x','y'])

# New_P=pd.DataFrame(columns=['t','p_p_X','p_p_Y'])

# current_t=df['t'][index]

x_p=np.zeros(4)
x_c=np.zeros(4)

#ошибки измерений (экспертное мнение)
Dphi=np.zeros((2,2))
Dphi[0][0]=4
Dphi[1][1]=4



#матрица наблюдений
# H=np.zeros((4,4))
# H[0][0]=1
# H[1][1]=1


p_p=np.zeros((4,4))
p_c=np.zeros((4,4))

p_p[0][0]=10
p_p[1][1]=10
p_p[2][2]=10
p_p[3][3]=10

x_p[0]=800
x_p[1]=200
x_p[2]=1
x_p[3]=1
# special_t=0

# while(1):

#     if ((t+dt)>current_t):
#         x_p,p_p=Prediction(df['t'][index],x_c,p_c)
#         Y=np.asarray([df['x'][index],df['y'][index]])
#         x_c,p_c=Correction(Y,df['t'][index],x_p,p_p)

#         New_X.loc[index]={'t':t,'x':x_c[0],'y':x_c[1],}
#         New_P.loc[index]={'t':t,'p_p_X':p_c[0][0],'p_p_Y':p_c[1][1],}
#         index=index+1
#         current_t=df['t'][index]+current_t
#         special_t=0
#         if (index==9): break

#     t=t+dt
#     special_t=special_t+dt
    
# New_X.to_csv('X2')
# New_P.to_csv('P2')