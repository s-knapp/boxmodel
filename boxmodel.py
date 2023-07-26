# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:11:59 2022

@author: Scott
"""

# 4 box model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

##########################################################
################### PARAMETERS ###########################
##########################################################

timesteps=6000*200 #second number is total years
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
M4=8e15 #8e15 #m3, vol of box 4
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
co2 = [4] #w/m2 4~2xCO2, 8~4xCO2, 12~8xCO2

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

#init temps for seasonal SW NOT REDONE WITH MSE
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

def annualmean(tseries,yearsteps=6000):
    means=[]
    for i in range( int( len(tseries)/yearsteps) ):
        means.append( np.mean(tseries[i*yearsteps:(i+1)*yearsteps]) )
    
    return np.asarray(means)

def monmean(tseries,monsteps=500):
    means=[]
    for i in range( int( len(tseries)/monsteps) ):
        means.append( np.mean(tseries[i*monsteps:(i+1)*monsteps]) )
    
    return np.asarray(means)

# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016GL071930
# ends up being about 4 w/m2 per doubling
def co2f(ppm):
    co2_0= 267 #ppm PI/reference
    a=-2.4e-7
    b=7.2e-4
    c=-2.1e-4
    
    forcing = ( a*(ppm-co2_0)**2 + b*(abs(ppm-co2_0))
    +c*1800 + 5.36 ) * np.log(ppm/co2_0)
    
    return forcing

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
#feedback with west influence: [box1,(box2,box1-box2),(box3,box1-box3)]
# fb from gradient seems to have lagged effect
# for box1-box2, lag is 1 month
# for box1-box3, lag is 4 months
#fb=[
#    [-4.0,(-3.0,-1.2),(-1.0,-0.5)]
#    ]

# agcm fixes ocean temps at initial and transport at 0
# slab ocean has free temps and transport fixed at equilibrium vals
agcm = False
slab = False
#%% MAIN LOOP

for c in range(len(co2)): #for each co2 forcing
    

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
    
            #fb param in west only depends on local temp
            fb1 = (T1 - t01) * fb[i][0]
            
            #seasonal SW cycle
    #        S1=SW[0][0][t]
            
    #        if i==0:
    #            R= S1*(1-alpha1) - (B*(T1-273.15) + A) + gamma*(Tmean-T1) + co2 + fb1 #+ nino[0][0][t]
    #        else:
                #once equil is found, fix longwave to that and let feedbacks account for change
                
            atm_div = mse_gamma*(mse_mean - mse(T1)) + mse_int #old version gamma*(Tmean-T1)
            R= S1*(1-alpha1) - (B*(t01-273.15) + A) + atm_div + co2[c] + fb1 #+ nino[0][0][t]  #use t01 for fixed OLR
                
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
            
            if not fb_mult: #single regional feedback parameter
                #local feedback: (Tnow - T0) * lambda
                fb2 = (T2 - t02) * fb[i][1]
            else:
                #feedback which includes W Pac, coefs from ceres.py mult regression
                # EastTOAanom = -4.0*EastSSTanom - 1.2*(WestSSTanom-EastSSTanom)
                fb2 = (T2 - t02) * fb[i][1][0] + ((T1-t01) - (T2-t02)) * fb[i][1][1]
            
            
            #seasonal SW cycle
    #        S2=SW[1][0][t]
            
    #        if i==0:
    #            R= S2*(1-alpha2) - (B*(T2-273.15) + A) + gamma*(Tmean-T2) + co2 + fb2 #+ nino[1][0][t]
    #        else:
                #once equil is found, fix longwave to that and let feedbacks account for change
            atm_div = mse_gamma*(mse_mean - mse(T2)) + mse_int #old version gamma*(Tmean-T2)
            R= S2*(1-alpha2) - (B*(t02-273.15) + A) + atm_div + co2[c] + fb2 #+ nino[1][0][t]
                
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
            R= S3*(1-alpha3) - (B*(t03-273.15) + A) + atm_div + co2[c] + fb3 #+ nino[2][0][t]
            
            sw[2].append(S3*(1-alpha3)); lw[2].append(B*(T3-273.15) + A)
            div[2].append(atm_div)
            circ[2].append(q*epsilon*(T2-T3) + q*(1-epsilon)*(T1-T3))
            H3= 1/(Cp*rho*h3) * R
            R3.append(R)
        #    H3OLR.append(H3olr)
        #    H3Latent.append(H3latent)
        
        
            
            
            if agcm:
                #no ocean transport!
                ocean1 = 0
                ocean2 = 0
                ocean3 = 0
                ocean4 = 0
                #fixed ocean temps
                T1=t01
                T2=t02
                T3=t03
                T4=t04
            elif slab:
                #fixed ocean transport at equilibrium values
                ocean1 = -36227431
                ocean2 = -164416748
                ocean3 = 200642990
                ocean4 = 1189
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
                
            if (t+1)%500==0: #write out data every 1/12 of a year
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

#%% write out data
    
#np.save('C:/Users/Scott/Documents/Python Scripts/boxmodel/fb.by1.4wm2.meanT.200.npy',np.asarray(meanT))
#np.save('C:/Users/Scott/Documents/Python Scripts/boxmodel/fb.by1.4wm2.allT.200.npy',np.asarray(allT))

#%% #########################################################################################
################ PLOTS - EVERYTHING BELOW HERE ARE VARIOUS PLOTS ############################
#############################################################################################

exps=[str(fb[i]) for i in range(len(fb))]
cmap = plt.get_cmap('jet')
expcols=cmap(np.linspace(0,1,len(fb)))#
expcols=['tab:blue','tab:orange','tab:green','tab:purple']

plt.figure(0)
expnum=len(allT)
for i in range(expnum):
    plt.plot(allT[i][0]-allT[i][1],color=expcols[i]) # T1 - T2
plt.title('Eq Gradient T1-T2')
plt.ylabel('T1-T2')
plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Years') 
plt.legend(['6e15','8e15','12e15'])

############
plt.figure(1)
plt.title("Temperatures")
expnum=len(allT)
for i in range(1):
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

meanT=[]
meanT.append( np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/fb.by1.4wm2.meanT.200.npy') )
meanT.append( np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/fb.by1.8wm2.meanT.npy') )
meanT.append( np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/fb.by1.16wm2.meanT.npy') )
allT=[]
allT.append( np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/fb.by1.4wm2.allT.200.npy') )
allT.append( np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/fb.by1.8wm2.allT.npy') )
allT.append( np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/fb.by1.16wm2.allT.npy') )

#meanTann = np.asarray( [annualmean(meanT[i,:]) for i in range(len(meanT))] )
#allTann = np.zeros(shape=(allT.shape[0],4,int(years)))
#for box in range(4):
#    allTann[:,box,:] = np.asarray( [annualmean(allT[i,box,:]) for i in range(len(allT))] )
#    
meanTmon=[]
for i in range(len(meanT)):
    if i==0: #fpr new longer files
        meanTmon.append( meanT[0])
    else:
        meanTmon.append( np.asarray( [monmean(meanT[i][j,:]) for j in range(len(meanT[i]))] ) )
allTmon = []
for i in range(len(meanT)):
    if i==0:
        allTmon.append(allT[0])
    else:
        allTmon.append( np.zeros(shape=(allT[i].shape[0],4,int(50*12))) )
        for box in range(4):
            allTmon[i][:,box,:] = np.asarray( [monmean(allT[i][j,box,:]) for j in range(len(allT[i]))] )
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

values = np.asarray( [meanT[0][i][-1] for i in range(len(fb))] )

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

#%% plot last timestep of all zonal gradients in all experiments
# as function of merid grad at last timestep
import statsmodels.api as sm
from scipy import stats

marks=['o','*','^']
cols = ['tab:blue','tab:orange','tab:red']


    
for i in range(216):
    x=[]
    y=[]
    for exp in range(3):
        zgradient = allT[exp][i][0][-1] - allT[exp][i][1][-1]
        mgradient = (allT[exp][i][0][-1] + allT[exp][i][1][-1])/2 - allT[exp][i][2][-1]
        y.append(zgradient)
        x.append(mgradient)
        plt.scatter(mgradient,zgradient,color=cols[exp],marker=marks[exp])
        
    #linear regression to determine slope for each fb
    x0 = sm.add_constant(x) #add column for intercept
    
    lr = sm.OLS(y,x0,missing='drop').fit()
    slope = lr.params[1]
    
    #slope from https://www.nature.com/articles/ngeo2577 determined to be 0.75+-0.07
    if (slope >= 0.68) and (slope <= 0.82):
        plt.plot(x,y,color='black',alpha=0.5,label=fb[i])
    else:
        plt.plot(x,y,color='grey',alpha=0.5)
    
plt.plot(np.linspace(4,8),0.75*np.linspace(4,8) )      
plt.ylabel('Zonal Gradient',fontsize=14)
plt.xlabel('Meridional gradient',fontsize=14)
plt.legend()
    
plt.title('Zonal vs Meridional gradients after 50 yrs    lines connect same feedback params')
#%% plot last timestep of all zonal gradients in all experiments
# as function of merid grad at last timestep SINGLE FEEDBACK VERSION
import statsmodels.api as sm
from scipy import stats

marks=['o','*','^']
cols = ['tab:blue','tab:orange','tab:red']


    

x=[]
y=[]
for exp in range(3):
    zgradient = allTmon[exp][:][0][-1] - allTmon[exp][:][1][-1]
    mgradient = (allTmon[exp][:][0][-1] + allTmon[exp][:][1][-1])/2 - allTmon[exp][:][2][-1]
    y.append(zgradient)
    x.append(mgradient)
    plt.scatter(mgradient,zgradient,color=cols[exp],marker=marks[exp])
    
#linear regression to determine slope for each fb
x0 = sm.add_constant(x) #add column for intercept

lr = sm.OLS(y,x0,missing='drop').fit()
slope = lr.params[1]

#slope from https://www.nature.com/articles/ngeo2577 determined to be 0.75+-0.07
if (slope >= 0.68) and (slope <= 0.82):
    plt.plot(x,y,color='black',alpha=0.5,label=fb[0])
else:
    plt.plot(x,y,color='grey',alpha=0.5)
    
plt.plot(np.linspace(np.min(x),np.max(x)),0.75*np.linspace(4,8) )      
plt.ylabel('Zonal Gradient',fontsize=14)
plt.xlabel('Meridional gradient',fontsize=14)
plt.legend()
    
plt.title('Zonal vs Meridional gradients after 50 yrs    lines connect same feedback params')

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

exp=2
values = np.asarray( [meanT[exp][i][-1] for i in range(len(fb))] )



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
    gradient = allT[exp][k][0] - allT[exp][k][1]
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
# lines colored by meanT at time
# feedback params of three boxes given as color of scatters at end of line

from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

exp=0 #co2 amount, 0,1,2=2x,4x,8x

zonalgrad = [ allTmon[exp][i][0] - allTmon[exp][i][1] for i in range(len(allTmon[exp])) ]
meridgrad = [ (allTmon[exp][i][0]+allTmon[exp][i][1])/2 - allTmon[exp][i][2] for i in range(len(allTmon[exp])) ]

fig = plt.figure(figsize=(7,7))
axs = fig.add_subplot()

#normalize to the min and max of all data, continuous colorbar
#norm = plt.Normalize(np.min(meanTmon), np.max(meanTmon))

#discrete colorbar for line color
cmap = plt.cm.get_cmap('plasma', 15)
norm = BoundaryNorm(np.logspace(
        np.log10( 1.0), np.log10(20),num=15), #np.max(meanTmon[exp])
            15)

#discrete colorbar for feedbacks
cmapfb = plt.cm.get_cmap('bwr', 6)
normfb = BoundaryNorm(np.arange(-4,2,1), 6)

for i in range(216):
    
    #this is a method for coloring the line by dividing it into
    #segments, each of which is colored individually
    #by providing a universal colorbar scale from the "norm" above,
    #all lines will use the same colorbar
    points = np.array([ zonalgrad[i], meridgrad[i] ]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm,alpha=0.8)
    # Set the values used for colormapping
    lc.set_array(meanTmon[exp][i,1:])
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    
    #plot markers indicating the feedback param in each box
    # many reach equil and bunch up all points, while some don't, need way to consistently assign scatters
    spacing=0.1
    x1 = zonalgrad[i][-1]
    x2i = np.argmin( np.abs(zonalgrad[i][:] - (x1 + spacing) ) )
    x2 = zonalgrad[i][x2i]
    x3i = np.argmin( np.abs(zonalgrad[i][:] - (x2 + spacing) ) )
    x3 = zonalgrad[i][x3i]
    y1 = meridgrad[i][-1]
    y2 = meridgrad[i][x2i]
    y3 = meridgrad[i][x3i]
    plt.scatter( x1, y1, marker="s", c=fb[i][0], 
                cmap=cmapfb, norm=normfb, zorder=10, alpha=0.8 )
    plt.scatter( x2, y2, marker="^", c=fb[i][1], 
                cmap=cmapfb, norm=normfb, zorder=10, alpha=0.8 )
    plt.scatter( x3, y3, marker="o", c=fb[i][2], 
                cmap=cmapfb, norm=normfb, zorder=10, alpha=0.8 )
 

axs.set_xlim((0.3,3.5))  
axs.set_ylim((5.9,7.2))
fig.colorbar(line, ax=axs,label='mean T')   
axs.set_xlabel('zonal gradient',fontsize=14)
axs.set_ylabel('meridional gradient',fontsize=14)
axs.set_facecolor('grey')

#plot * at initial value (same for all)
x0 = allT[exp][i][0][0] - allT[exp][i][1][0]
y0 = (allT[exp][i][0][0]+allT[exp][i][1][0])/2 - allT[exp][i][2][0]
plt.scatter(x0, y0, marker='*',s=50,color='red',zorder=10)

# draw lines through initial point to divide into quadrants
plt.hlines(y0, xmin = 0.3, xmax= 3.5, linestyle = '--', color = 'black', alpha=0.7)
plt.vlines(x0, ymin = 5.9, ymax= 7.2, linestyle = '--', color = 'black', alpha=0.7)


legend_elements = [Line2D([0], [0], marker='s', lw=0, label='Box 1',
                          markerfacecolor='black', markersize=10),
                   Line2D([0], [0], marker='^', lw=0, label='Box 2',
                          markerfacecolor='black', markersize=10),
                   Line2D([0], [0], marker='o', lw=0, label='Box 3',
                          markerfacecolor='black', markersize=10)
                   ]
axs.legend(handles=legend_elements, loc='upper right')


#fig.savefig('gradsandtemp.png')
#plt.close(fig)
#%% HISTOGRAM OF FB PARAMS IN FOUR QUADRANTS FROM ABOVE PLOT
# BUT ACTUALLY COLORED GRIDS OF RELATIVE FB PARAM STRENGTHS ZONAL/MERID
# Notes about results of this plot:
# quad 1: more pos box 2 warms box 2, weakens zonal, while more neg box 3 cools box 3, strengthens merid
# quad 2: more pos box 1 warms box 1, strengthen zonal, while more neg box 3 cools box 3, strengthens merid
# quad 3: more neg box 1 cools box 1, weakens zonal, while more pos box 3 warms box 3, weakens merid
# quad 4: more neg box 2 cools box 2, strengthen zonal, more pos box 1 warms box 1, also strengthen zonal

zonalgrad = [ allTmon[exp][i][0] - allTmon[exp][i][1] for i in range(len(allTmon[exp])) ]
meridgrad = [ (allTmon[exp][i][0]+allTmon[exp][i][1])/2 - allTmon[exp][i][2] for i in range(len(allTmon[exp])) ]

x0 = allT[exp][0][0][0] - allT[exp][0][1][0]
y0 = (allT[exp][0][0][0]+allT[exp][0][1][0])/2 - allT[exp][0][2][0]

#quad 1: x<x0, y>y0. quad 2: x>x0, y>y0. quad 3: x<x0, y<y0. quad 4: x>x0, y<y0
titles=['Zonal weaken & Merid strengthen','Zonal strengthen & Merid strengthen',
        'Zonal weaken & Merid weaken','Zonal strengthen & Merid weaken']
quads = [ [], [], [], [] ]
for i in range(len(zonalgrad)):
    if (zonalgrad[i][-1] < x0) & (meridgrad[i][-1] > y0):
        quads[0].append(fb[i])
    elif (zonalgrad[i][-1] > x0) & (meridgrad[i][-1] > y0):
        quads[1].append(fb[i])
    elif (zonalgrad[i][-1] < x0) & (meridgrad[i][-1] < y0):
        quads[2].append(fb[i])
    elif (zonalgrad[i][-1] > x0) & (meridgrad[i][-1] < y0):
        quads[3].append(fb[i])

fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(8,7))
ax = ax.flatten()
for i in range(4): # for each quadrant
    #value of each fb param in quad[i]
    fb1quad = [ quads[i][j][0] for j in range(len(quads[i]))]
    fb2quad = [ quads[i][j][1] for j in range(len(quads[i]))]
    fb3quad = [ quads[i][j][2] for j in range(len(quads[i]))]
    
    #create stats about relative param strengths, 9 possibilities
    arr = np.zeros((3,3))
#    zonalstr=0 ; zonaleq=0 ; zonalweak=0  
#    meridstr = 0 ; merideq=0 ; meridweak = 0
    for j in range(len(fb1quad)):
        if (fb1quad[j] > fb2quad[j]): # if box 1 param > box 2 param, col 1
            if fb3quad[j] < np.mean([fb1quad[j],fb2quad[j]]): #extrop<trop row 1
                arr[2,0] +=1
            elif fb3quad[j] == np.mean([fb1quad[j],fb2quad[j]]): #row 2
                arr[1,0] +=1
            else: #row 3
                arr[0,0] +=1
        elif fb1quad[j] == fb2quad[j]: #col 2
            if fb3quad[j] < np.mean([fb1quad[j],fb2quad[j]]): #extrop<trop
                arr[2,1] +=1
            elif fb3quad[j] == np.mean([fb1quad[j],fb2quad[j]]):
                arr[1,1] +=1
            else:
                arr[0,1] +=1
        elif (fb1quad[j] < fb2quad[j]): # if box 3 param > avg of box1&2 col 3
            if fb3quad[j] < np.mean([fb1quad[j],fb2quad[j]]): #extrop<trop
                arr[2,2] +=1
            elif fb3quad[j] == np.mean([fb1quad[j],fb2quad[j]]):
                arr[1,2] +=1
            else:
                arr[0,2] +=1
   
    arr = arr/len(fb1quad) * 100
    
    # create discrete colormap
    cmap = plt.cm.get_cmap('Reds', 10)
    norm = BoundaryNorm(np.linspace(0,100,10), ncolors=10)
    
    # plot colored grid
    ax[i].imshow(arr, cmap=cmap,norm=norm)
    
    # draw gridlines
    #ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax[i].set_xticks(np.arange(0, 3, 1))
    ax[i].set_xticklabels(['West>East','West=East','West<East'])
    ax[i].set_yticks(np.arange(0, 3, 1))
    ax[i].set_yticklabels(['ExTrop>Trop','ExTrop=Trop','ExTrop<Trop'])
    
    #add text of number in each grid
    for row in range(3):
        for col in range(3):
            ax[i].annotate(f"{arr[row,col].round(decimals=1)}%", (col-0.15,row+0.05),weight='bold')
        
        
#    pltx = np.arange(-4,2)
#    ax[i].bar(pltx-0.1,np.histogram(fb1quad,bins=6,range=(-4,1))[0], label='box 1',width=0.1)
#    ax[i].bar(pltx, np.histogram(fb2quad,bins=6,range=(-4,1))[0], label='box 2',width=0.1)
#    ax[i].bar(pltx+0.1, np.histogram(fb3quad,bins=6,range=(-4,1))[0], label='box 3',width=0.1)
#    
#    ax[i].annotate(f"Zonal Params Strengthen = {np.round(zonalstr/len(fb1quad),decimals=1) * 100}%",(-2.5,28))
#    ax[i].annotate(f"Zonal Params Weaken = {np.round(zonalweak/len(fb1quad),decimals=1) * 100}%",(-2.5,26.5))
#    ax[i].annotate(f"Merid Params Strengthen = {np.round(meridstr/len(fb1quad),decimals=1) * 100}%",(-2.5,25))
#    ax[i].annotate(f"Merid Params Weaken = {np.round(meridweak/len(fb1quad),decimals=1) * 100}%",(-2.5,23.5))
#    
#    if i==0:
#        ax[i].legend()
    ax[i].set_title(titles[i])
#plt.tight_layout()
#    ax[i].set_ylim([0,31])
#fig.suptitle('R')
#%% equil mean T as function of zonal/merid grad
    
zonalgrad = [ allTmon[exp][i][0] - allTmon[exp][i][1] for i in range(len(allTmon[exp])) ]
meridgrad = [ (allTmon[exp][i][0]+allTmon[exp][i][1])/2 - allTmon[exp][i][2] for i in range(len(allTmon[exp])) ]

#for i in range(len(zonalgrad)):
#    plt.scatter( zonalgrad[i][-1], meanTmon[0][i][-1])
#plt.xlim( (0,4) )
#plt.ylim( (0,20) )

#plt.figure()
#for i in range(len(meridgrad)):
#    plt.scatter( meridgrad[i][-1], meanTmon[0][i][-1])
#plt.xlim( (4.5,7.6) )
#plt.ylim( (0,20) )

#there are some clear linear-ish structures in here

cmap = plt.get_cmap('rainbow')(np.linspace(0,1, 36))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
c=0
six=0
#empty lists for regression
x1=[]
x2=[]
y=[]
#draw a line through each set as well
for i in range(len(zonalgrad)-41): #excluding some of the later fb combos that explode and ruin the plot
    if c==6: #each pair of box1/2 params has 6 vals of box 3, so after 6 do OLS and reset
        #OLS
        X = np.column_stack((x1,x2,np.ones(len(x1))))
        y=np.asarray(y)
        coefs = np.linalg.lstsq(X,y,rcond=None)[0]
#        z_pred = coefs[0]*np.linspace(0,4,50) + coefs[1]*np.linspace(4.5,7.5,50)+coefs[2]
        z_pred=coefs[0]*np.linspace(min(X[:,0]),max(X[:,0]),50) + coefs[1]*np.linspace(min(X[:,1]),max(X[:,1]),50) + coefs[2]
        ax.plot(np.linspace(min(X[:,0]),max(X[:,0]),50), np.linspace(min(X[:,1]),max(X[:,1]),50),z_pred,color=cmap[six])
#        ax.plot(X[:,0], X[:,1],z_pred,color=cmap[six])
        
        #reset for new pairs
        six+=1
        c=0
        x1=[]
        x2=[]
        y=[]
    #append data for OLS
    x1.append(zonalgrad[i][-1])
    x2.append(meridgrad[i][-1])
    y.append(meanTmon[0][i][-1])
    
    ax.scatter( zonalgrad[i][-1], meridgrad[i][-1], meanTmon[0][i][-1], color=cmap[six])
    c+=1
    
    if i==( len(zonalgrad)-1): #if at last step, do OLS and plot
        X = np.column_stack((x1,x2,np.ones(len(x1))))
        y=np.asarray(y)
        coefs = np.linalg.lstsq(X,y,rcond=None)[0]
        z_pred = coefs[0]*np.linspace(0,4,50) + coefs[1]*np.linspace(4.5,7.5,50)+coefs[2]
        ax.plot(np.linspace(0,4,50), np.linspace(4.5,7.5,50),z_pred,color=cmap[six])
    
ax.set_zlim([0,20])
ax.set_xlim([0,4])
ax.set_ylim([4.5,7.5])
ax.set_xlabel('Zonal grad')
ax.set_ylabel('Meridional grad')
ax.set_zlabel('Mean T')
    


#%% ENERGY BALANCE - FIXED SW/LW + CO2 + FB*T
# MAKE DATAFRAME OF CHARACTERISTICS OF GRADIENTS/SENSITIVITIES

colors = ['tab:blue','tab:orange','tab:red']
sens = []
co2 = [4]

#create dataframe with feedbacks, clim sens, and net feedback (estimated from regression)
df = pd.DataFrame({'Box Feedbacks':fb, 'Sensitivity':np.zeros(len(fb)), 
                   'Net Feedback':np.zeros(len(fb)), 'Max Gradient':np.zeros(len(fb)),
                   'Time of Max Grad':np.zeros(len(fb)), 'Init Gradient':np.zeros(len(fb)),
                   'Final Gradient':np.zeros(len(fb)), 'Time of Equal Grad':np.zeros(len(fb))})
    
dfm = pd.DataFrame({'Box Feedbacks':fb, 'Sensitivity':np.zeros(len(fb)), 
                   'Net Feedback':np.zeros(len(fb)), 'Max Gradient':np.zeros(len(fb)),
                   'Time of Max Grad':np.zeros(len(fb)), 'Init Gradient':np.zeros(len(fb)),
                   'Final Gradient':np.zeros(len(fb)), 'Time of Equal Grad':np.zeros(len(fb))})

for c in range(len(co2)):
    sens.append([])
    zonalgrad = [ allTmon[c][i][0] - allTmon[c][i][1] for i in range(len(allTmon[c])) ]
    meridgrad = [ (allTmon[c][i][0]+allTmon[c][i][1])/2 - allTmon[c][i][2] for i in range(len(allTmon[c])) ]
    for i in range(len(fb)):
        
        T1 = allTmon[c][i][0]
        T2 = allTmon[c][i][1]
        T3 = allTmon[c][i][2]
        
        
        toa_b1 = S1*(1-alpha1) - (B*(t01-273.15) + A) + co2[c] + (T1 - t01) * fb[i][0]
        toa_b1 = toa_b1*area1 / (area1+area2+area3)
        toa_b2 = S2*(1-alpha2) - (B*(t02-273.15) + A) + co2[c] + (T2 - t02) * fb[i][1]
        toa_b2 = toa_b2*area2 / (area1+area2+area3)
        toa_b3 = S3*(1-alpha3) - (B*(t03-273.15) + A) + co2[c] + (T3 - t03) * fb[i][2]
        toa_b3 = toa_b3*area3 / (area1+area2+area3)
        
        #linear regression to determine x intercept for each fb (=clim sens)
        # netTOA = slope*meanT + y-int
        # climsens = - y-int/slope
        x0 = sm.add_constant(meanTmon[c][i]) #add column for intercept
        
        lr = sm.OLS(toa_b1+toa_b2+toa_b3,x0,missing='drop').fit()
        yint = lr.params[0]
        slope = lr.params[1]
        cs=-yint/slope
        sens[c].append( cs )

        #add things to zonal dataframe
        if not (cs>0) & (cs<50):
            df['Sensitivity'][i] = np.nan
        else:
            df['Sensitivity'][i] = cs
        df['Net Feedback'][i] = slope
        df['Max Gradient'][i] = np.max(zonalgrad[i])
        df['Init Gradient'][i] = zonalgrad[i][0]
        df['Final Gradient'][i] = zonalgrad[i][-1]
        df['Time of Max Grad'][i] = np.argmax(zonalgrad[i])
        #thermosat duration 
        thermdur = np.argmin(np.abs(zonalgrad[i][0] - zonalgrad[i][5:])) + 5
        if thermdur == 5: #if gradient only increases in strength
            df['Time of Equal Grad'][i] =  np.nan
        else:
            df['Time of Equal Grad'][i] = thermdur
        
        #add things to meridional dataframe
        if not (cs>0) & (cs<50):
            dfm['Sensitivity'][i] = np.nan
        else:
            dfm['Sensitivity'][i] = cs
        dfm['Net Feedback'][i] = slope
        dfm['Max Gradient'][i] = np.max(meridgrad[i])
        dfm['Init Gradient'][i] = meridgrad[i][0]
        dfm['Final Gradient'][i] = meridgrad[i][-1]
        dfm['Time of Max Grad'][i] = np.argmax(meridgrad[i])
        dfm['Time of Equal Grad'][i] = np.argmin(np.abs(meridgrad[i][0] - meridgrad[i][5:])) + 5 
        
        
        #plt.scatter( meanTmon[c][i], toa_b1+toa_b2+toa_b3 ,color = colors[c], s=5)
        #plt.scatter( sens[c][i],0)
        #add regression line to visualize x intercept
        #plt.plot( np.arange(0,30), slope*np.arange(30)+yint, color='grey',linewidth=0.5,alpha=0.6)

#plt.xlabel('Mean T',fontsize=14)
#plt.ylabel('Net Energy Imbalance (+ down)',fontsize=14)
#plt.ylim([0,35])
#plt.xlim([0,30])
#sens = np.asarray(sens)
#ecs = np.zeros_like(sens)
#
#
#for i in range(sens.shape[0]):
#    ecs[i,:] = sens[i,:]/(co2[i]/2**i)
#
#for i in range(len(fb)):
##for c in range(len(co2)):
##    plt.scatter( np.arange(216), sens[c,:], color=colors[c], s=5)
#    plt.plot(sens[:,i])
#plt.xticks(ticks=np.arange(len(co2)))
#    
#plt.ylim([0,20])
#
#plt.xlabel('Experiment')
#plt.ylabel('ECS [C]')
#%%
# PLOTS USING THE DATAFRAME

# y as function of box feedback params
# select only rows with data

# notes: max gradient - controlled solely by box 1
#        time of max gradient - controlled mostly by box 1
#        sensitivity is controlled mostly by box 3, likely due to large size
#        net feedback mostly controlled by box 3, definitely due to size
#        time to return to init grad is sensitive to outliers!
#        final gradient controlled mostly by box 1, could be due to unrealistic pos params though


var = 'Final Gradient'
isdata = np.where(df[var].notnull())[0]
marks=['o','x','^']
offset=[-0.1,0.,0.1]


#plot 3 times
for f in range(3):
    
    #individual regional fb params
#    x = np.asarray( [ i[f] for i in df['Box Feedbacks'].iloc[isdata][:] ] )
#    labels=['fb1','fb2','fb3']
    #differences of regional fb params
    x = np.asarray( [ i[f]-i[f-1] for i in df['Box Feedbacks'].iloc[isdata][:] ] )
    labels=['fb1-fb3','fb2-fb1','fb3-fb2']
    y = df[var].iloc[isdata]
    
    #regression to determine linear fit
    x0 = sm.add_constant(x) #add column for intercept
            
    lr = sm.OLS(y,x0,missing='drop').fit()
    
    plt.scatter( x+offset[f], y , marker=marks[f], label = f"{labels[f]} {lr.rsquared.round(decimals=3)}")
    
#meridional gradient as predictor
#x = np.asarray( [ (i[0]+i[1])/2 - i[2] for i in df['Box Feedbacks'].iloc[isdata][:] ] )
#y = df[var].iloc[isdata]

#regression to determine linear fit
#x0 = sm.add_constant(x) #add column for intercept
#        
#lr = sm.OLS(y,x0,missing='drop').fit()
#
#plt.scatter( x, y , label = f"(fb1+fb2)/2 - fb3 {lr.rsquared.round(decimals=3)}")
    
plt.ylabel(var,fontsize=13)
plt.xlabel('Feedback parameter',fontsize=13)
plt.legend()
#%% calculate effective clim sens (clim sens as estimated via lin reg at the current time)
###  


colors = ['tab:blue','tab:orange','tab:red']
sens = []
co2 = [4]

#for each co2 forcing
for c in range(len(co2)): 
    
    zonalgrad = [ allTmon[c][i][0] - allTmon[c][i][1] for i in range(len(allTmon[exp])) ]
    effcs = np.zeros( (allTmon[0].shape[0],allTmon[0].shape[2]) ) #effective cs for each fb, at each time
    
    for f in range(len(fb)): # for each feedback
    
        for t in range(2, allTmon[0].shape[2]): # for each timestep starting at 2nd
            
            T1 = allTmon[c][f,0,:t]
            T2 = allTmon[c][f,1,:t]
            T3 = allTmon[c][f,2,:t]
            
            
            toa_b1 = S1*(1-alpha1) - (B*(t01-273.15) + A) + co2[c] + (T1 - t01) * fb[f][0]
            toa_b1 = toa_b1*area1 / (area1+area2+area3)
            toa_b2 = S2*(1-alpha2) - (B*(t02-273.15) + A) + co2[c] + (T2 - t02) * fb[f][1]
            toa_b2 = toa_b2*area2 / (area1+area2+area3)
            toa_b3 = S3*(1-alpha3) - (B*(t03-273.15) + A) + co2[c] + (T3 - t03) * fb[f][2]
            toa_b3 = toa_b3*area3 / (area1+area2+area3)
            
            #linear regression to determine x intercept for each fb (=clim sens)
            # netTOA = slope*meanT + y-int
            # climsens = - y-int/slope
            x0 = sm.add_constant(meanTmon[c][f][:t]) #add column for intercept
            
            lr = sm.OLS(toa_b1+toa_b2+toa_b3,x0,missing='drop').fit()
            yint = lr.params[0]
            slope = lr.params[1]
            
            effcs[f,t] = -yint/slope
 
#%% plot eff cs as function of sst gradient
colors=['tab:blue','tab:orange','tab:green','tab:red']
            
#choose specific fb            
for f in range(len(fb)):
    #if  (fb[f][0]<-2) & (fb[f][1]==fb[f][0]) & (fb[f][2]>-2) &( (np.max(effcs[f,:])<20) & (np.max(effcs[f,:])>0) ):   
    if (fb[f][0]==-4):
        plt.scatter( zonalgrad[f], effcs[f,:],s=3,label=fb[f],color=colors[0])
    if (fb[f][0]==-3):
        plt.scatter( zonalgrad[f], effcs[f,:],s=3,label=fb[f],color=colors[1])
    if (fb[f][0]==-2):
        plt.scatter( zonalgrad[f], effcs[f,:],s=3,label=fb[f],color=colors[2])
    if (fb[f][0]==-1):
        plt.scatter( zonalgrad[f], effcs[f,:],s=3,label=fb[f],color=colors[3])
        #plt.scatter( zonalgrad[f][::12], effcs[f,::12],marker='*')

#plt.legend()
plt.xlabel('Zonal SST Gradient', fontsize=14)
plt.ylabel('effective equilibrium CS',fontsize=14)  
plt.ylim([0,30])     
plt.xlim([1.5,2.75]) 

#legend_elements = [Line2D([0], [0], marker='s', lw=0, label='Box 1',
#                          markerfacecolor='black', markersize=10),
#                   Line2D([0], [0], marker='^', lw=0, label='Box 2',
#                          markerfacecolor='black', markersize=10),
#                   Line2D([0], [0], marker='o', lw=0, label='Box 3',
#                          markerfacecolor='black', markersize=10)
#                   ]
#axs.legend(handles=legend_elements, loc='upper right')
        
    
    
#%% meridional vs zonal T gradient
# lines colored by effCS at time
# feedback params of three boxes given as color of scatters at end of line

from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

exp=0 #co2 amount, 0,1,2=2x,4x,8x

zonalgrad = [ allTmon[exp][i][0] - allTmon[exp][i][1] for i in range(len(allTmon[exp])) ]
meridgrad = [ (allTmon[exp][i][0]+allTmon[exp][i][1])/2 - allTmon[exp][i][2] for i in range(len(allTmon[exp])) ]

fig = plt.figure(figsize=(7,7))
axs = fig.add_subplot()

#normalize to the min and max of all data, continuous colorbar
#norm = plt.Normalize(np.min(meanTmon), np.max(meanTmon))

#discrete colorbar for line color
cmap = plt.cm.get_cmap('plasma', 15)
norm = BoundaryNorm(np.linspace(
        1, 20,num=15), 
            15)

#discrete colorbar for feedbacks
cmapfb = plt.cm.get_cmap('bwr', 6)
normfb = BoundaryNorm(np.arange(-4,2,1), 6)

for i in range(216):
    
    #this is a method for coloring the line by dividing it into
    #segments, each of which is colored individually
    #by providing a universal colorbar scale from the "norm" above,
    #all lines will use the same colorbar
    points = np.array([ zonalgrad[i], meridgrad[i] ]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm,alpha=0.8)
    # Set the values used for colormapping
    lc.set_array(effcs[i,1:])
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    
    #plot markers indicating the feedback param in each box
    plt.scatter( zonalgrad[i][-1], meridgrad[i][-1], marker="s", c=fb[i][0], 
                cmap=cmapfb, norm=normfb, zorder=10, alpha=0.8 )
    plt.scatter( zonalgrad[i][-40], meridgrad[i][-40], marker="^", c=fb[i][1], 
                cmap=cmapfb, norm=normfb, zorder=10, alpha=0.8 )
    plt.scatter( zonalgrad[i][-80], meridgrad[i][-80], marker="o", c=fb[i][2], 
                cmap=cmapfb, norm=normfb, zorder=10, alpha=0.8 )
 

#axs.set_xlim((0.5,3.5))  
#axs.set_ylim((5.9,7.1))
fig.colorbar(line, ax=axs,label='eff CS')   
axs.set_xlabel('zonal gradient',fontsize=14)
axs.set_ylabel('meridional gradient',fontsize=14)
axs.set_facecolor('grey')

plt.scatter(allT[exp][i][0][0] - allT[exp][i][1][0],(allT[exp][i][0][0]+allT[exp][i][1][0])/2 - allT[exp][i][2][0],
                marker='*',s=50,color='red',zorder=10)



legend_elements = [Line2D([0], [0], marker='s', lw=0, label='Box 1',
                          markerfacecolor='black', markersize=10),
                   Line2D([0], [0], marker='^', lw=0, label='Box 2',
                          markerfacecolor='black', markersize=10),
                   Line2D([0], [0], marker='o', lw=0, label='Box 3',
                          markerfacecolor='black', markersize=10)
                   ]
axs.legend(handles=legend_elements, loc='upper right')

axs.set_ylim([6.3,7.05])
axs.set_xlim([1.8,3])
