# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:11:59 2022

@author: Scott
"""

# 4 box model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##########################################################
################### PARAMETERS ###########################
##########################################################

timesteps=6000*50 #second number is total years
dt=5256 #seconds, 1/6000 of year
print(f"timestep dt ={dt/60:.2f} minutes")
totaltime=timesteps*dt
years=totaltime/3600/24/365
print(f"total time = {years:.2f} years")
Cp = 4.186e3 # J/(kg K) specific heat of ocean water
rho = 1027 # kg/m3 density of sea water at surface
rhoair = 1.225 # kg/m3 dens of air
p_air=100000 # Pa air pressure

epsilon = 0.48 # ratio of flow out
M1=6.8e14 #m3, vol of box 1
M2=6.8e14 #m3, vol of box 2
M3=4e15 #m3, vol of box 3
M4=8e15 #m3, vol of box 4
Aw=1e6  #m3/(s K) walker coupling param
Ah=4e6  #m3/(s K) hadley coupling param
h1=50 #m, depth of box 1
h2=50 #m, depth of box 2
h3=200 #m, depth of box 3

#constants for MSE div
cp_air=1005; #J/kg/K
L=2250000; #J/kg or 2260000
ps=100000; #Pa
RH=0.8; # relative humidity
epsil=0.622; # ration of mass or vapour to dry air
mse_gamma = 0.001581 # W/m2 / J/kg
mse_int = -9.56 #intercept for mse regression
#regression done on Cheyenne notebook mse_aht_regression.ipynb
#used only 65S-65N

#params from Burls et al 2014 atmos EBM
# shortwave in = S(1-alpha)
# TOA OLR = BT + A
# atmos convergence = gamma(Tmean - T)
B = 1.7 # W/m2 / K
A = 214 # W/m2
gamma = 3.3 # W/m2 / K

#CO2 FORCING AS A CONSTANT EVERYWHERE
co2 = 0 #w/m2

#the following based on CERES data calculated in ceres.py
#to find range of alphas, use extreme values from all ceres data?
alpha1 = 0.26; alpha2 = 0.20; alpha3 = 0.36
S1 = 415; S2 = 415; S3 = 318

#ranges of SW from CERES
# S1 and S2: 386 in june to 437 in march
S1min=386; S1max=437
# S3: 143 in dec to 481 in june
S3min=143; S3max=481

#create timeseries of SW for each 

SW=[[],[],[]]
for i in range(3):
    if i in [0,1]:
        smin=S1min
        smax=S1max
        shift=np.pi/2
    else:
        smin=S3min
        smax=S3max
        shift=np.pi
    
    cyc=(smax-smin)/2 *np.cos(np.linspace(0-shift,2*np.pi-shift,6000)) + smin + (smax-smin)/2
    cyc=np.tile(cyc,int(years))
    SW[i].append(cyc)
    
#plt.plot(SW[2][0])
    
#use nino data as forcing?
n3=pd.read_table("D:/ninoindices/nina3.anom.data",sep="\s+",header=1,index_col=0,skipfooter=3,dtype=np.float32)

nino=[[],[],[]]
for i in range(3):
    if i==1: #east
        dat=np.asarray(n3.loc[np.arange(1980,2001)])
        dat=dat.flatten()
        
        #find slope
        X=np.vstack([np.arange(0,len(dat)),np.ones(len(dat))]).T
        m,c=np.linalg.lstsq(X,dat,rcond=None)[0]
        print(f"i={i} slope is {m}")
        
        dat=np.repeat(dat,500) #repeat monthly values for each timestep per month (500)
        nino[i].append(dat)
    else:
        nino[i].append(np.zeros(timesteps))
        
#init temps for fixed SW and MSE divergence
t01=299.12 #init T box 1 WEST TROPICS
t02=296.75 #init T box 2 EAST TROPICS
t03=291.16 #init T box 3 EX TROPICS/Nor Pacific
t04=291.16 #init T box 4 EQ UNDER CURRENT
        
#init temps for fixed SW and gamma*(Tmean-T)
#t01=306.34 #init T box 1 WEST TROPICS
#t02=303.45 #init T box 2 EAST TROPICS
#t03=297.91 #init T box 3 EX TROPICS/Nor Pacific
#t04=297.91 #init T box 4 EQ UNDER CURRENT

#init temps for seasonal SW
#t01=304.1 #init T box 1 WEST TROPICS
#t02=301.3 #init T box 2 EAST TROPICS
#t03=296.0 #init T box 3 EX TROPICS/Nor Pacific
#t04=296.0 #init T box 4 EQ UNDER CURRENT

area1=M1/h1 #vol/depth
area2=M2/h2
area3=M3/h3

totvol = M1+M2+M3
weights=[M1/totvol,M2/totvol,M3/totvol]


#array for temps from all experiments, is appended automatically for each new exp
allT=[]
meanT=[]
allLatent=[]
allR=[]
ocean=[]

#%%
###########################################################
###################FUNCTIONS#################################
###########################################################

def es(t):
    t-=273.15
    p=0.61094*np.exp(17.625*t/(t+243.04)) #kPa
    p=p*1000 #Pa
    return p

def mse(t): #moist static energy, constant rel hum, surface pressure
    t=t-273.15
    sat_vap_pressure = 1000.0 * 0.6112 * np.exp(17.67 * t / (t + 243.5)); # T in Celcius
    qs=epsil/ps*sat_vap_pressure; #kg/kg
    h=cp_air*(t+273.15)+L*RH*qs; #moist static energy J/kg
    return h

    
# sigmoid curve based on T gradient. plateus at lower and upper values
    #need a stable region near t1-t2=3 ? add two curves?
def logi(t1,t2):
    base=1/(1+np.exp(t2-t1+3))
    mod=(base)*0.05 +0.27
    return mod

##############################################################
#####################EXECUTION################################
##############################################################


#iterate through feedback params
#feedbacks as only change of radiation, S and longwave fixed
#first do only regional feedbacks
# then add gradient dependency as well
    
#order: [box1,box2,box3]
fb=[
#    [0,0,0]
#    [-5,-0.1,0.5],
#    [-1,-5,-4]
    ]

#loop to make fb iterations
# total combinations of n numbers in sequence r long is n**r

step = 1
fb_low = -4
fb_high = 1

for i in np.arange(fb_low,fb_high+step,step):
    for j in np.arange(fb_low,fb_high+step,step):
        for k in np.arange(fb_low,fb_high+step,step):
            fb.append([i,j,k])

#fb=[[0,0,0]]

fb_mult=False #make this true to use the mult feedbacks below
#feedback with west influence: [box1,(box2,box1),(box3,box1)]
#fb=[
#    [(0.88,1.24,-1.04),(-3.13,-5.94),(1.43,-1.46)]
#    ]

#agcm fixes ocean temps and transport at start values
agcm = True
#%% MAIN LOOP

for i in range(len(fb)): #for each feedback set
    
    t1=[t01]
    t2=[t02]
    t3=[t03]
    t4=[t04]
    R1=[]; R2=[]; R3=[]
    H1OLR=[]; H2OLR=[]; H3OLR=[]
    H1Latent=[]; H2Latent=[]; H3Latent=[]
    sw=[[],[],[]]; lw=[[],[],[]]; div=[[],[],[]]
    circ=[[],[],[]]
    ot=[[],[],[],[]]

    Tmean0 = 1/np.sum(weights)*(t1[0]*weights[0] + t2[0]*weights[1] + t3[0]*weights[2])
    
    for t in range(timesteps-1):
        
        
        #define initial temps
        if t==0: 
            T1=t01
            T2=t02
            T3=t03
            T4=t04
            dT1=0 ; dT2=0; dT3=0
            
        
        q=Ah*( np.average([T1,T2],weights=[M1,M2]) - T3) + Aw*(T1-T2) 
        
        
        Tmean = 1/np.sum(weights)*(T1*weights[0] + T2*weights[1] + T3*weights[2])
        mse_mean = 1/np.sum(weights)*(mse(T1)*weights[0] + mse(T2)*weights[1] + mse(T3)*weights[2])

        if not fb_mult:
            #local feedback: (Tnow - T0) * lambda
            fb1 = (T1 - t01) * fb[i][0]
        else:
            #feedback which includes W Pac, coefs from ceres.py mult regression
            # WestTOAanom = 2.33*WestSSTanom - 1.36*EastSSTanom
            fb1 = (T1 - t01) * fb[i][0][0] + ((T1 - t01) - (T2 - t02)) * fb[i][0][1] + ((T1 - t01) - (T3-t03)) * fb[i][0][2]
    
        #seasonal SW cycle
#        S1=SW[0][0][t]
        
#        if i==0:
#            R= S1*(1-alpha1) - (B*(T1-273.15) + A) + gamma*(Tmean-T1) + co2 + fb1 #+ nino[0][0][t]
#        else:
            #once equil is found, fix longwave to that and let feedbacks account for change
            
        atm_div = mse_gamma*(mse_mean - mse(T1)) + mse_int #old version gamma*(Tmean-T1)
        R= S1*(1-alpha1) - (B*(t01-273.15) + A) + atm_div + co2 + fb1 #+ nino[0][0][t]  #use t01 for fixed OLR
            
        sw[0].append(S1*(1-alpha1)); lw[0].append(B*(T1-273.15) + A)
        div[0].append(atm_div)
        circ[0].append(q*(1-epsilon)*(T2-T1))
        H1= 1/(Cp*rho*h1) * R
        R1.append(R)
    #    H1OLR.append(H1olr)
        
        
        # east Pac feedback via Walker weakening
        # albedo (alpha) logistic function of T1-T2...?
        # 
    #    alpha2 =  logi(T1,T2)
        
        if not fb_mult:
            #local feedback: (Tnow - T0) * lambda
            fb2 = (T2 - t02) * fb[i][1]
        else:
            #feedback which includes W Pac, coefs from ceres.py mult regression
            # EastTOAanom = -3.13*EastSSTanom - 5.94*(WestSSTanom-EastSSTanom)
            fb2 = (T2 - t02) * fb[i][1][0] + ((T1-t01) - (T2-t02)) * fb[i][1][1]
        
        
        #seasonal SW cycle
#        S2=SW[1][0][t]
        
#        if i==0:
#            R= S2*(1-alpha2) - (B*(T2-273.15) + A) + gamma*(Tmean-T2) + co2 + fb2 #+ nino[1][0][t]
#        else:
            #once equil is found, fix longwave to that and let feedbacks account for change
        atm_div = mse_gamma*(mse_mean - mse(T2)) + mse_int #old version gamma*(Tmean-T2)
        R= S2*(1-alpha2) - (B*(t02-273.15) + A) + atm_div + co2 + fb2 #+ nino[1][0][t]
            
        sw[1].append(S2*(1-alpha2)); lw[1].append(B*(T2-273.15) + A)
        div[1].append(atm_div)
        circ[1].append(q*(T4-T2))
        H2= 1/(Cp*rho*h2) * R
        R2.append(R)
    #    H2OLR.append(H2olr)
        
        if not fb_mult:
            #local feedback: (Tnow - T0) * lambda
            fb3 = (T3 - t03) * fb[i][2]
        else:
            #feedback which includes W Pac, coefs from ceres.py mult regression
            # NorthTOAanom = 1.43*NorthSSTanom - 1.46*(WestSSTanom-NorthSSTanom)
            fb3 = (T3 - t03) * fb[i][2][0] + ((T1-t01) - (T3-t03)) * fb[i][2][1]
        
        #seasonal SW cycle
#        S3=SW[2][0][t]
        
#        if i==0:
#            R= S3*(1-alpha3) - (B*(T3-273.15) + A) + gamma*(Tmean-T3) + co2 + fb3 #+ nino[2][0][t]
#        else:
            #once equil is found, fix longwave to that and let feedbacks account for change
        atm_div = mse_gamma*(mse_mean - mse(T3)) + mse_int #old version gamma*(Tmean-T3)
        R= S3*(1-alpha3) - (B*(t03-273.15) + A) + atm_div + co2 + fb3 #+ nino[2][0][t]
        
        sw[2].append(S3*(1-alpha3)); lw[2].append(B*(T3-273.15) + A)
        div[2].append(atm_div)
        circ[2].append(q*epsilon*(T2-T3) + q*(1-epsilon)*(T1-T3))
        H3= 1/(Cp*rho*h3) * R
        R3.append(R)
    #    H3OLR.append(H3olr)
    #    H3Latent.append(H3latent)
    
    
        
        
        if agcm:
            #fixed ocean transport at equilibrium values
            ocean1 = -36227431
            ocean2 = -164416748
            ocean3 = 200642990
            ocean4 = 1189
            #fixed ocean temps
            T1=t01
            T2=t02
            T3=t03
            T4=t04
        else:
            #ocean transport
            ocean1 = q*(1-epsilon)*(T2-T1)
            ocean2 = q*(T4-T2)
            ocean3 = q*epsilon*(T2-T3) + q*(1-epsilon)*(T1-T3)
            ocean4 = q*(T3-T4)
            #normal temperature evolution equations
            T1=T1 + dt/M1 * (M1*H1 + ocean1) 
            T2=T2 + dt/M2 * (M2*H2 + ocean2 )#+ M2*nino[1][0][t-1]/(3600*24*30))
            T3=T3 + dt/M3 * (M3*H3 + ocean3) 
            T4=T4 + dt/M4 * ocean4
        
        dT1=T1-t1[0]
        dT2=T2-t2[0]
        dT3=T3-t3[0]
            
        #if (t+1)%3000==0: #write out data twice a year
        t1.append(T1)
        t2.append(T2)
        t3.append(T3)
        t4.append(T4)
        
        ot[0].append(ocean1)
        ot[1].append(ocean2)
        ot[2].append(ocean3)
        ot[3].append(ocean4)
                
    t1=np.asarray(t1)
    t2=np.asarray(t2)
    t3=np.asarray(t3)
    t4=np.asarray(t4)
    
    ocean.append( np.asarray(ot) )
    
    allT.append([t1,t2,t3,t4])
    allR.append([R1,R2,R3])
    
#####################################
# sensitivity

#totvol = M1+M2+M3
#weights=[M1/totvol,M2/totvol,M3/totvol]
#
    Tmean = 1/np.sum(weights)*(t1*weights[0] + t2*weights[1] + t3*weights[2])
    Tmean0 = 1/np.sum(weights)*(t1[0]*weights[0] + t2[0]*weights[1] + t3[0]*weights[2])
    Tmeandiff = Tmean - Tmean0
    
    meanT.append(np.round(Tmeandiff,decimals=2))
#lam_eff = (1/Tmeandiff)*(lamdaTW*T1 + lamdaTE*T2 + lamdaET*T3)/3
#Teff= -R/lam_eff     

#%% PLOTS

exps=[str(fb[i]) for i in range(len(fb))]
cmap = plt.get_cmap('jet')
expcols=cmap(np.linspace(0,1,len(fb)))#['tab:blue','tab:orange','tab:green','tab:purple']

plt.figure(0)
expnum=len(allT)
for i in range(expnum):
    plt.plot(allT[i][0]-allT[i][1],color=expcols[i]) # T1 - T2
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
plt.annotate("Ex Trop",xy=(24000,allT[0][2][24000]))
plt.annotate("UnderCurrent",xy=(0,allT[0][3][24000]))

plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Years') 
plt.legend()
#
#
#############
#plt.figure(2)
#plt.title('Radiation Balance')
#plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
#plt.xlabel('Years') 
#expnum=len(allR)
#for i in range(expnum):
#    plt.plot(allR[i][0],color=expcols[i],label=exps[i])
#    plt.plot(allR[i][1],color=expcols[i])
#    plt.plot(allR[i][2],color=expcols[i])
#    
#plt.annotate("Trop W",xy=(0,allR[0][0][0]-0.3))
#plt.annotate("Trop E",xy=(0,allR[0][1][0]-0.3))
#plt.annotate("Ex Trop",xy=(0,allR[0][2][0]))
#
#plt.legend()
#
#############
#plt.figure(3)
#plt.title("Mean Temp Change")
#expnum=len(meanT)
#for i in range(expnum):
#    plt.plot(meanT[i],color=expcols[i],label=exps[i])
#    
#plt.ylabel('K')    
#plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
#plt.xlabel('Years') 
#plt.legend()
#
#############
##only works for last exp right now
plt.figure(4)
plt.title('Atm Div')
plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Years') 
expnum=len(allR)
for i in range(expnum):
    plt.plot(div[0],color=expcols[i],label=exps[i])
    plt.plot(div[1],color=expcols[i])
    plt.plot(div[2],color=expcols[i])
    
plt.annotate("Trop W",xy=(0,div[0][0]-0.3))
plt.annotate("Trop E",xy=(0,div[1][0]-0.3))
plt.annotate("Ex Trop",xy=(0,div[2][0]))

plt.legend()

#plt.figure(5,figsize=(12,3))
#
#for i in range(3):
#    plt.subplot(2,2,1)
#    plt.title('sw toa')
#    plt.plot(sw[i],label=f"sw box{i+1}",color=expcols[i])
#    plt.legend()
#    plt.subplot(2,2,2)
#    plt.title('lw toa')
#    plt.plot(lw[i],label=f"lw box{i+1}",color=expcols[i])
#    plt.legend()
#    plt.subplot(2,2,3)
#    plt.title('atm div')
#    plt.plot(div[i],label=f"div box{i+1}",color=expcols[i])
#    plt.legend()
#    plt.subplot(2,2,4)
#    plt.plot(circ[i],label=f"ocean circ into box{i+1}", color=expcols[i])
#    plt.legend()

plt.figure(6)
plt.title("Ocean Transport")
expnum=len(ocean)
for i in range(expnum):
    plt.plot(ocean[i][0],color=expcols[i],label=exps[i])
    plt.plot(ocean[i][1],color=expcols[i])
    plt.plot(ocean[i][2],color=expcols[i])
    plt.plot(ocean[i][3],color=expcols[i])

plt.annotate("Trop W",xy=(0,ocean[0][0][0]-0.3))
plt.annotate("Trop E",xy=(0,ocean[0][1][0]-0.3))
plt.annotate("Ex Trop",xy=(24000,ocean[0][2][24000]))
plt.annotate("UnderCurrent",xy=(0,ocean[0][3][24000]))

plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Years') 
plt.legend()

#def logi(t1,t2):
#    base=1/(1+np.exp(t2-t1+3))
#    mod=(base)*0.05 +0.27
#    return mod
#
#plt.plot(3-np.linspace(-4,6),logi(3,np.linspace(-4,6)))
#plt.xlabel('T1-T2')

#nino tiem series added as term

#%% OPEN AND PLOT SENSITIVITY RUNS

meanT= np.load('fb.by1.meanT.npy')
allT= np.load('fb.by1.allT.npy')

#import mpltern


#fig = plt.figure(figsize=(6, 7))
#ax = plt.subplot(projection='ternary')
#
##box1
#top = np.asarray( [fb[i][0]/np.sum(fb[i]) for i in range(len(fb))] ) 
#top[np.isnan(top)]=1/3 #nans caused by dividing by 0 set to 0
##box2
#left = np.asarray( [fb[i][1]/np.sum(fb[i]) for i in range(len(fb))] ) 
#left[np.isnan(left)]=1/3 #nans caused by dividing by 0 set to 0
##box3
#right = np.asarray( [fb[i][2]/np.sum(fb[i]) for i in range(len(fb))] ) 
#right[np.isnan(right)]=1/3 #nans caused by dividing by 0 set to 0
#
#values = np.asarray( [meanT[i][-1] for i in range(len(fb))] )
#cs = ax.tricontourf(top, left, right, values)
#
#fig.colorbar(cs)
#             
#ax.set_tlabel('Box 1')
#ax.set_llabel('Box 2')
#ax.set_rlabel('Box 3')
#westeast_westnorth = np.meshgrid( )
#plt.contourf( )
#%% Mean T as function of box feedback

#box1
fbb1 = np.asarray( [fb[i][0] for i in range(len(fb))] ) 
#box2
fbb2 = np.asarray( [fb[i][1] for i in range(len(fb))] ) 
#box3
fbb3 = np.asarray( [fb[i][2] for i in range(len(fb))] ) 

values = np.asarray( [meanT[i][-1] for i in range(len(fb))] )

plt.scatter(fbb1-0.1,values,label='Box 1',marker='x')
plt.scatter(fbb2,values,label='Box 2',marker='x')
plt.scatter(fbb3+0.1,values,label='Box 3',marker='x')
plt.legend()
plt.xlabel('Local Feedback Param')
plt.ylabel('Mean warming after 50 years of +4 W/m2')

#%% box1-box2

for i in range(125):
    gradient = allT[i][0] - allT[i][1]
    
    if np.max(gradient) > gradient[-1]: #if max is greater than last temp, thermostat
        
        plt.scatter( fbb1[i] - fbb2[i] , 
                    gradient[-1] - gradient[0],
                    color='tab:blue',s=5)
        
    else: #if gradient is strongest at end, no thermostat
        plt.scatter( fbb1[i] - fbb2[i] , 
                    gradient[-1] - gradient[0],
                    color='tab:blue',s=5,facecolor='white')
    
plt.xlabel('box 1 fb - box 2 fb')
plt.ylabel('box 1 - box 2 yr50-yr0')
plt.hlines(0,xmin=-4,xmax=4,linestyle='--')

#%% plot all gradient time series

for i in range(125):
    gradient = allT[i][0] - allT[i][1]
    plt.plot(gradient,color='tab:blue')
    
#%% pick constant global feedback
    
def gmean_fb(box1fb,box2fb,box3fb,vol1,vol2,vol3):
    fbs=[box1fb,box2fb,box3fb]
    gmean = np.average( fbs, weights = [vol1,vol2,vol3] )
    return gmean

means=[]
for i in fb:
    means.append( gmean_fb(i[0],i[1],i[2],M1,M2,M3) )
    
#plt.hist(means,bins=(10))
    
binlims=np.linspace(np.min(fb),np.max(fb),11)

#box1
fbb1 = np.asarray( [fb[i][0] for i in range(len(fb))] ) 
#box2
fbb2 = np.asarray( [fb[i][1] for i in range(len(fb))] ) 
#box3
fbb3 = np.asarray( [fb[i][2] for i in range(len(fb))] ) 

values = np.asarray( [meanT[i][-1] for i in range(len(fb))] )

plt.figure(figsize=(20,7))
plt.title('Mean T as function of mean FB')
for k in range(len(fb)):
    
    for i in range(len(binlims)-1):
        binmin=binlims[i]
        binmax=binlims[i+1]
        
        if (means[k] > binmin) & (means[k] < binmax):
            plt.subplot(2,5,i+1)
            plt.scatter(fbb1[k]-0.1,values[k],label='Box 1',marker='x',color='tab:blue')
            plt.scatter(fbb2[k],values[k],label='Box 2',marker='x',color='tab:orange')
            plt.scatter(fbb3[k]+0.1,values[k],label='Box 3',marker='x',color='tab:green')
            plt.title(f"{binmin} - {binmax}")
            plt.grid()
            
            continue

plt.figure(figsize=(20,7))
plt.title('% W-E gradient change as function of mean FB')



for k in range(len(fb)):
    gradient = allT[k][0] - allT[k][1]
    gradchg= (gradient[-1] - gradient[0]) / gradient[0] * 100
    for i in range(len(binlims)-1):
        binmin=binlims[i]
        binmax=binlims[i+1]
        
        if (means[k] > binmin) & (means[k] < binmax):
            plt.subplot(2,5,i+1)
            plt.scatter(fbb1[k]-0.1,gradchg,label='Box 1',marker='x',color='tab:blue')
            plt.scatter(fbb2[k],gradchg,label='Box 2',marker='x',color='tab:orange')
            plt.scatter(fbb3[k]+0.1,gradchg,label='Box 3',marker='x',color='tab:green')
            plt.title(f"{binmin} - {binmax}")
            plt.grid()
            
            continue
        

#%% mean T as function of  fb param
            
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs = [fb[i][0] for i in range(np.shape(fb)[0])]
ys = [fb[i][1] for i in range(np.shape(fb)[0])]
zs = [fb[i][2] for i in range(np.shape(fb)[0])]
colors = [ meanT[i][-1] for i in range(len(meanT))]

cs = ax.scatter(xs, ys, zs, cmap='viridis', c=colors, vmin=0, vmax=12)

ax.set_xlabel('Box1 FB')
ax.set_ylabel('Box2 FB')
ax.set_zlabel('Box3 FB')

fig.colorbar(cs)
    

#%% meridional vs zonal T gradient

from matplotlib.collections import LineCollection
#from matplotlib.colors import ListedColormap, BoundaryNorm

zonalgrad = [ allT[i][0] - allT[i][1] for i in range(len(allT)) ]
meridgrad = [ (allT[i][0]+allT[i][1])/2 - allT[i][2] for i in range(len(allT)) ]


fig = plt.figure(figsize=(7,7))
axs = fig.add_subplot()

#normalize to the min and max of all data
norm = plt.Normalize(np.min(meanT), np.max(meanT))

for i in range(100):
    
    #this is a method for coloring the line by dividing it into
    #segments, each of which is colored individually
    #by providing a universal colorbar scale from the "norm" above,
    #all lines will use the same colorbar
    points = np.array([ zonalgrad[i], meridgrad[i] ]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(meanT[i,1:])
    lc.set_linewidth(1)
    line = axs.add_collection(lc)
    
axs.set_xlim((0.5,3.5))  
axs.set_ylim((5.5,7.5))
fig.colorbar(line, ax=axs)   
axs.set_xlabel('zonal gradient')
axs.set_ylabel('meridional gradient')
fig.savefig('gradsandtemp.png')
plt.close(fig)