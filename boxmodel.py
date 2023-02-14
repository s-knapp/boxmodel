# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:11:59 2022

@author: Scott
"""

# 4 box model

import numpy as np
import matplotlib.pyplot as plt

##########################################################
################### PARAMETERS ###########################
##########################################################

timesteps=6000*30 #second number is total years
dt=5256 #seconds, 1/6000 of year
print(f"timestep dt ={dt/60:.2f} minutes")
totaltime=timesteps*dt
years=totaltime/3600/24/365
print(f"total time = {years:.2f} years")
#rhoair*Cd*Va1*L*0.622/p_air*es(T1)*(1-RH)
Cp = 4.186e3 # J/(kg K) specific heat of ocean water
Cd = 8.8e-4 # dimensionless drag coefficient https://en.wikipedia.org/wiki/Drag_coefficient
rho = 1027 # kg/m3 density of sea water at surface
rhoair = 1.225 # kg/m3 dens of air
p_air=100000 # Pa air pressure

epsilon = 0.48 # ratio of flow out of box 2 into box 1
M1=6.8e14 #m3, vol of box 1
M2=6.8e14 #m3, vol of box 2
M3=4e15 #m3, vol of box 3
M4=8e15 #m3, vol of box 4
Aw=1e6  #m3/(s K) walker coupling param
Ah=4e6  #m3/(s K) hadley coupling param
h1=50 #m, depth of box 1
h2=50 #m, depth of box 2
h3=200 #m, depth of box 3
SwTW=240 #w/m2 sw forcing tropics West
SwTE=240 #w/m2 sw forcing tropics East
SwEx=140 #w/m2 sw forcing extropics
E=0.18 #emissivity for control
sigma=5.67e-8 #w/m2K4 stefan-botzmann constant
Va1= 5 #wind speed box 1
Va2= 5 #wind speed box 2
Va3= 5 #wind speed box 3
Ld= 1.5e-3 #latent heat coefficient
L= 2.260e6 #J/kg latent heat of evap
RH=0.6 #rel hum
HsensT= 10 #w/m2 sensible heat flux Tropics
HsensExT= 15 #w/m2 sensible heat flux exTropics

t01=303 #init T box 1 WEST TROPICS
t02=300 #init T box 2 EAST TROPICS
t03=292 #init T box 3 EX TROPICS
t04=296 #init T box 4 EQ UNDER CURRENT

#array for temps from all experiments, is appended automatically for each new exp
allT=[]
meanT=[]
allLatent=[]
allR=[]

#%%
###########################################################
###################SCHEMES#################################
###########################################################

#forward time
def ft1(T1,T2,dt,q,epsilon,M1,H1):
     
    tplus1 = T1 + dt/M1 * (M1*H1 + q*(1-epsilon)*(T2-T1)) 
    
    
    return tplus1

def ft2(T2,T4,dt,q,M2,H2):
     
    tplus1 = T2 + dt/M2 * (M2*H2 + q*(T4-T2)) 
    
    
    return tplus1

def ft3(T3,T1,T2,dt,q,epsilon,M3,H3):
     
    tplus1 = T3 + dt/M3 * (M3*H3 + q*epsilon*(T2-T3) + q*(1-epsilon)*(T1-T3)) 
    
    
    return tplus1

def ft4(T4,M4,q,T3):
    
    tplus1 = T4 + dt/M4 * q*(T3-T4)
    
    return tplus1


def es(t):
    t-=273.15
    p=0.61094*np.exp(17.625*t/(t+243.04)) #kPa
    p=p*1000 #Pa
    return p

#radiative feedbacks, W/m2

feedbacks=True
if feedbacks == False:
    lamdaTW=0
    lamdaTE=0
    lamdaET=0
else:
    lamdaTW=-10.0
    lamdaTE=5.0
    lamdaET=-0.5
    
    
#def es2(t):
#    t=t-273.15
#    es=np.exp(34.494- (4924.99/(t+237.1)))/(t+105)**(1.57)
#    return es
# from https://journals.ametsoc.org/view/journals/apme/57/6/jamc-d-17-0334.1.xml
# equation 17
##############################################################
#####################EXECUTION################################
##############################################################

t1=[t01]
t2=[t02]
t3=[t03]
t4=[t04]
R1=[]; R2=[]; R3=[]
H1OLR=[]; H2OLR=[]; H3OLR=[]
H1Latent=[]; H2Latent=[]; H3Latent=[]


#uniform warming
E=E*0.85


#%% MAIN LOOP
for t in range(timesteps-1):
    
    
    #define initial temps
    if t==0: 
        T1=t01
        T2=t02
        T3=t03
        T4=t04
        dT1=0 ; dT2=0; dT3=0
        
#    taux=0.01013145882*(T1-T2)
#    Va_1=np.sqrt(taux/(Cd*rhoair*10**-3))
#    
#    taux=0.01013145882*(T3 - 0.5*(T1-T2))
#    Va_et1=np.sqrt(taux/(Cd*rhoair*10**-3))
    
    #the taux version of q gives weird behavior, all temps converge
#    q=Va_1*Aw + Va_et1*Ah
    
    q=Ah*( np.average([T1,T2],weights=[M1,M2]) - T3) + Aw*(T1-T2) #np.average([T1,T2],weights=[M1,M2])
    
    H1latent=rhoair*Cd*Va1*L*0.622/p_air*es(T1)*(1-RH)
    H1olr=E*sigma*T1**4
    R=SwTW-H1latent-HsensT-H1olr +lamdaTW*dT1
    H1= 1/(Cp*rho*h1) * R
    R1.append(R)
    H1OLR.append(H1olr)
    H1Latent.append(H1latent)
    
    H2latent=rhoair*Cd*Va2*L*0.622/p_air*es(T2)*(1-RH)
    H2olr=E*sigma*T2**4
    R=SwTE-H2latent-HsensT-H2olr +lamdaTE*dT2
    H2= 1/(Cp*rho*h2) * R
    R2.append(R)
    H2OLR.append(H2olr)
    H2Latent.append(H2latent)
    
    H3latent=rhoair*Cd*Va3*L*0.622/p_air*es(T3)*(1-RH)
    H3olr=E*sigma*T3**4
    R=SwEx-H3latent-HsensExT-H3olr +lamdaET*dT3
    H3= 1/(Cp*rho*h3) * R
    R3.append(R)
    H3OLR.append(H3olr)
    H3Latent.append(H3latent)
    
    T1=ft1(T1,T2,dt,q,epsilon,M1,H1)
    T2=ft2(T2,T4,dt,q,M2,H2)
    T3=ft3(T3,T1,T2,dt,q,epsilon,M3,H3)
    T4=ft4(T4,M4,q,T3)
    
    dT1=T1-t1[0]
    dT2=T2-t2[0]
    dT3=T3-t3[0]
        
    #if (t+1)%3000==0: #write out data twice a year
    t1.append(T1)
    t2.append(T2)
    t3.append(T3)
    t4.append(T4)
            
t1=np.asarray(t1)
t2=np.asarray(t2)
t3=np.asarray(t3)
t4=np.asarray(t4)

allT.append([t1,t2,t3,t4])
allR.append([R1,R2,R3])
    
#####################################
# sensitivity
#area1=M1/h1 #vol/depth
#area2=M2/h2
#area3=M3/h3
totvol = M1+M2+M3
weights=[M1/totvol,M2/totvol,M3/totvol]

Tmean = 1/np.sum(weights)*(t1*weights[0] + t2*weights[1] + t3*weights[2])
Tmean0 = 1/np.sum(weights)*(t1[0]*weights[0] + t2[0]*weights[1] + t3[0]*weights[2])
Tmeandiff = Tmean - Tmean0

meanT.append(np.round(Tmeandiff,decimals=2))
lam_eff = (1/Tmeandiff)*(lamdaTW*T1 + lamdaTE*T2 + lamdaET*T3)/3
#Teff= -R/lam_eff     

#%% PLOTS

exps=['No feedbacks','Feedbacks']
expcols=['tab:blue','tab:orange','tab:green','tab:purple']

plt.figure(0)
expnum=len(allT)
for i in range(expnum):
    plt.plot(allT[i][0]-allT[i][1]) # T1 - T2
plt.title('Eq Gradient T1-T2')
plt.ylabel('T1-T2')
plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Years') 
plt.legend(exps)

############
plt.figure(1)
plt.title("Temperatures")
expnum=len(allT)
for i in range(expnum):
    plt.plot(allT[i][0],color=expcols[i],label=exps[i])
    plt.plot(allT[i][1],color=expcols[i])
    plt.plot(allT[i][2],color=expcols[i])
    plt.plot(allT[i][3],color=expcols[i])

plt.annotate("Trop W",xy=(0,allT[0][0][0]-0.3))
plt.annotate("Trop E",xy=(0,allT[0][1][0]-0.3))
plt.annotate("Ex Trop",xy=(0,allT[0][2][0]))
plt.annotate("UnderCurrent",xy=(0,allT[0][3][0]))

plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Years') 
plt.legend()


############
plt.figure(2)
plt.title('Radiation Balance')
plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Years') 
expnum=len(allR)
for i in range(expnum):
    plt.plot(allR[i][0],color=expcols[i],label=exps[i])
    plt.plot(allR[i][1],color=expcols[i])
    plt.plot(allR[i][2],color=expcols[i])
    
plt.annotate("Trop W",xy=(0,allR[0][0][0]-0.3))
plt.annotate("Trop E",xy=(0,allR[0][1][0]-0.3))
plt.annotate("Ex Trop",xy=(0,allR[0][2][0]))

plt.legend()
