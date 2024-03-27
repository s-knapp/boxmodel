# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:11:59 2022

@author: Scott
"""

# 4...now 5.....now 7! box model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

##########################################################
################### PARAMETERS ###########################
##########################################################

timesteps=6000*150 #second number is total years
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
warea = 15774945387831 #surface area of box 1, west
earea = 15774945387831
narea = 33306750009756
sarea = 28541763159964
hnarea = 12430460163633
hsarea = 20244594358318
h1=50 #60 #m, depth of box 1
h2=50 #20 #50 #m, depth of box 2
h3=200 #m, depth of box 3
h4=200 #m, depth of box 5
h6=500 # m, depth of box 6
h7=500 # m, depth of box 7
M1= warea * h1 #7.8e14 #4.9e14 #3e14 6.8e14 #m3, vol of box 1 (west)
M2= earea * h2 #3.2e14 #7.8e14 #4.9e14 #3e14#6.8e14 #m3, vol of box 2 (east)
M3= narea * h3 #6.6e15 #4.5e15 #m3, vol of box 3  (north) 
M4= sarea * h4 #5.7e15 #4.3e15 #m3, vol of box 4 (south)
#for 20-50N 160-240E =24031081172332.445*200=4806216234466489.0 ~ 4.8e15m3
#was using 4e15
underarea = warea + earea + narea + sarea
underdepth = 150 #150 for cmip-like...?
M5=underarea * underdepth #9.5e15 #4e15 #9.6e15 #8e15 #m3, vol of box 5 (undercurrent)

Aw=1e6 #0.1e6  #m3/(s K) walker coupling param
Ah=4e6 #2.5e6 #4e6  #m3/(s K) hadley coupling param


#constants for MSE div
cp_air=1005; #J/kg/K
L=2250000; #J/kg or 2260000
ps=100000; #Pa
RH=0.8; # relative humidity in tropics
RHet=0.7 # relative humidity in ex tropics? probably not, looks the same
epsil=0.622; # ration of mass or vapour to dry air
mse_gamma = 0.001581 # W/m2 / J/kg
mse_gamma_w = -0.007 #
mse_gamma_e = -0.002 #
mse_gamma_n = -0.002 #
mse_gamma_s = -0.005 #
mse_gamma_hn = -0.002
mse_gamma_hs = -0.003
mse_int = -9.56 #-9.56 #intercept for mse regression
mse_int_w = 75 #118.72 
mse_int_e = -3 #17.72 
mse_int_n = -45 #-25.94
mse_int_s = -29 #-9.32 
mse_int_hn = 6 #13.93
mse_int_hs = -36 #157.55
#regression done on Cheyenne notebook mse_aht_regression.ipynb
#used only 65S-65N

#params from Burls et al 2014 atmos EBM
# shortwave in = S(1-alpha)
# TOA OLR = BT + A
B = 1.7 # W/m2 / C
Bwest = -16.73 #-0.33
Beast = -6.80 #-2.79
Bnorth = 2.10 #-0.98
Bsouth = 2.74 #-4.2

A = 214 # W/m2
Awest = 729.92 #214.11
Aeast = 453.8 #346.18
Anorth = 206.1 #259.78
Asouth = 202.12 #348.36

# atmos convergence = gamma(Tmean - T)
gamma = 3.3 # W/m2 / K

#the following based on CERES data calculated in ceres.py
#to find range of alphas, use extreme values from all ceres data?
#boxes used for ceres radiation data: 
# 1: 8-8N/S 120-200E  2: 8-8N/S 200-280E   3: 8-50N 165-230E 4: 8-40S 200-280E
alpha1 = 0.23; alpha2 = 0.20; alpha3 = 0.25; alpha4 = 0.23 
alpha6 = 0.45 # High lats NH
alpha7 = 0.43 # High lats SH
S1 = 415; S2 = 415; S3 = 367; S4 = 380
S6 = 228 # High lats NH
S7 = 254 # High lats SH

LW1 = 238 #LW out from CERES data of box
LW2 = 273
LW3 = 248
LW4 = 256
LW6 = 202
LW7 = 214

# ERA5 SW+LW
swlw1 = 73
swlw2 = 55
swlw3 = 18
swlw4 = 26
swlw6 = -81
swlw7 = -59

#ranges of SW from CERES
# S1 and S2: 386 in july to 437 in april
S1min=386; S1max=437
# S3: 200 in dec to 475 in july
S3min=200; S3max=475
# S4: 234 in june to 498 in jan
S4min=234; S4max=498
#create timeseries of SW for each 
#or read them in from ceres and interp between monthly data points
ceres_ann = np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/ceres.annual.cycles.npy')
SW=[[],[],[],[]]
for i in range(4):
#    if i in [0,1]: #tropics
#        smin=S1min
#        smax=S1max
#        shift=np.pi/2
#    elif i==2: #northern hemisphere
#        smin=S3min
#        smax=S3max
#        shift=np.pi
#    else: #southern
#        smin=S4min
#        smax=S4max
#        shift=0
    
    #make one annual cycle
#    cyc=(smax-smin)/2 *np.cos(np.linspace(0-shift,2*np.pi-shift,6000)) + smin + (smax-smin)/2
#    #repeat annual cycle for number of years
    #interp between ceres points to get SW at each timestep
    cyc = np.interp(np.linspace(0,12,6000), np.arange(13), ceres_ann[i][:13])
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


area1=M1/h1 #vol/depth
area2=M2/h2
area3=M3/h3
area4=M4/h4

totvol = M1+M2+M3+M4
weights=[M1/totvol,M2/totvol,M3/totvol,M4/totvol]


#%%
###########################################################
###################FUNCTIONS#################################
###########################################################

def linchange(y1,y2,x1change,x2change, x):
    # create line between two points, (x1change,y1) and (x2change,y2)
    # returns a function which will return y for a given x
    slope = (y2-y1)/(x2change-x1change)
    intercept = y1 - slope*x1change
    
    
    return slope*x + intercept

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

def mse_et(t): #moist static energy in extratropics, constant rel hum, surface pressure
    t=t-273.15
    sat_vap_pressure = 1000.0 * 0.6112 * np.exp(17.67 * t / (t + 243.5)); # T in Celcius
    qs=epsil/ps*sat_vap_pressure; #kg/kg
    h=cp_air*(t+273.15)+L*RHet*qs; #moist static energy J/kg
    return h

def aht_anom( avglat_here, avglat_poleward , 
             mse_here, mse_poleward):
    #estimate poleward AHT anom from meridional changes in MSE anoms
    # from https://journals.ametsoc.org/view/journals/clim/32/12/jcli-d-18-0563.1.xml?tab_body=fulltext-display
    ps = 100000 # surf press Pa
    D = 0.96e6 #diffusivity const
    x = np.sin( np.deg2rad(avglat_here) )
    xp = np.sin( np.deg2rad(avglat_poleward ) )
    dx = xp - x
    dh = mse_poleward - mse_here
    F = -(2*np.pi*ps) * D * (1-x**2) * dh/dx
    
    return F
    
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

def rmean(var,interval):
	if interval %2 == 0:
    		print("INTERVAL NEEDS TO BE ODD")
    		return
	l=[]
	buffer=int((interval-1)/2)
	for i in range(buffer,len(var)-buffer):
    		l.append(np.mean(var[i-buffer:i+buffer]) )
    
	#for easy plotting, add nan values to make length same as original var
	for i in range(buffer):
    		l.insert(0,np.nan)
    		l.append(np.nan)
	return np.asarray(l)

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

def calcmean(allt): #had a bug in meanT calc after adding 5th box, this manually recalcs
    t1=allt[0]; t2=allt[1]; t3=allt[2]; t4=allt[3]
    Tmean = 1/np.sum(weights)*(t1*weights[0] + t2*weights[1] + t3*weights[2]+ t4*weights[3])
    Tmean0 = 1/np.sum(weights)*(t1[0]*weights[0] + t2[0]*weights[1] + t3[0]*weights[2] + t4[0]*weights[3])
    Tmeandiff = Tmean - Tmean0
    
    return np.round(Tmeandiff,decimals=2)
#%%
##############################################################
#####################SETUP################################
##############################################################


#iterate through feedback params
#feedbacks as only change of radiation, S and longwave fixed
#first do only regional feedbacks
# then add gradient dependency as well
    

#fb from kernel method
fb=[
    [-1.520469842017107, -1.2556147562653937, -2.1034417111556967, -1.745977415054225],
[-1.1400492464071237, -0.8066724197518209, -1.832457305057804, -1.6517927220561486],
[-1.041983066946665, -1.0784420260781429, -1.902692957091249, -1.378516651980788],
[-1.0636575763372529, -1.0462062948010409, -1.8431624892952088, -1.5007292562736254],
[-1.3247319314695214, -1.071377727089415, -2.1937409918234905, -1.7010246603303718],
[-1.1927125188137766, -0.7075954438821985, -2.0524353402546107, -1.5725998303085396],
[-1.1809698040064736, -0.7322345342383156, -2.088096889591821, -1.5489817703329976],
[-1.1353132234121024, -0.6174384378552894, -2.058900708062625, -1.5284073574847308],
[-1.115076198118857, -0.809461179828845, -2.128702501806357, -1.636195328932258],
[-0.9421016856909648, -0.5624049036874643, -1.933930282340918, -1.498575660053308],
[-1.0316057157285448, -0.590485131423812, -1.933654651304624, -1.4837049235646282],
[-1.4821837005120793, -1.0894088265836002, -1.9894286394108127, -1.7541668660817886],
[-1.0208175039109504, -0.7932807922847116, -2.002259970193138, -1.6763584885143383],
[-0.8884099274655746, -0.6019268949078367, -2.0505567262034243, -1.6076073897485577],
[-0.7469880964962712, -0.5491591130942219, -2.028422622835825, -1.462259220184485],
[-0.9716245942519643, -0.6477459737389316, -2.021801166620142, -1.766905492481839],
[-1.1983621899360344, -0.5001179160464191, -2.1832751008385074, -1.5948894556726],
[-0.9911023268206997, -1.0515216516379402, -1.8065918383468476, -1.562233772533083],
[-0.9691726791731288, -0.9486838822702068, -1.712607584029826, -1.4647114776026258],
[-1.1452567351653211, -1.2073843890586446, -1.948033718868184, -1.5702703383947365]
    ]

mnames= ['ACCESS-CM2', 'AWI-CM-1-1-MR', 'BCC-CSM2-MR', 'BCC-ESM1', 'CAMS-CSM1-0', 
         'CESM2', 'CESM2-FV2', 'CESM2-WACCM', 'CESM2-WACCM-FV2', 'GISS-E2-1-G', 
         'GISS-E2-2-G', 'HadGEM3-GC31-MM', 'MIROC-ES2L', 'MIROC6', 'MPI-ESM1-2-HR', 
         'MPI-ESM1-2-LR', 'NorESM2-MM', 'SAM0-UNICON', 'TaiESM1', 'UKESM1-0-LL']

mcolors = ['red', 'red', 'blue', 'blue', 'blue', 'red', 'red', 'red', 'red', 'blue', 
           'blue', 'red', 'blue', 'blue', 'blue', 'blue', 'blue', 'red', 'red', 'red']

#### generate sensitivity based on cmip fb ranges with 4 values for each region #####
# cfb = []
# r1 = [ fb[i][0] for i in range(len(fb)) ] #fb param of first box
# r2 = [ fb[i][1] for i in range(len(fb)) ]
# r3 = [ fb[i][2] for i in range(len(fb)) ]
# r4 = [ fb[i][3] for i in range(len(fb)) ]
# buffer = 0.5 #extra value to add to min/max of each range

# for i in np.linspace(min(r1) - buffer, max(r1) + buffer, 4):
#     for j in np.linspace(min(r2) - buffer, max(r2) + buffer, 4):
#         for k in np.linspace(min(r3) - buffer, max(r3) + buffer, 4):
#             for l in np.linspace(min(r4) - buffer, max(r4) + buffer, 4):
#                 cfb.append([i,j,k,l]) 
#                
#fb = cfb
####################
#loop to make fb iterations
# total combinations of n numbers in sequence r long is n**r
# 6**3 = 216, 5**4 = 625

# fb = []

# step = 1
# fb_low = -2
# fb_high = 2

# for i in np.arange(fb_low,fb_high+step,step):
#     for j in np.arange(fb_low,fb_high+step,step):
#         for k in np.arange(fb_low,fb_high+step,step):
#             # fb.append([i,j,k,k]) #using same param for north/south
#             for s in np.arange(fb_low,fb_high+step,step):
#                 fb.append([i,j,k,s])

#non uniform fb range
# fb = []

# fbrange = [-2,-1,1,2]

# for i in fbrange:
#     for j in fbrange:
#         for k in fbrange:
#             # fb.append([i,j,k,k]) #using same param for north/south
#             for s in fbrange:
#                 fb.append([i,j,k,s])
                
###########################

#cmip6 mean fb
# fb = [ np.mean( np.asarray(fb), axis=0 ) ]
# fb = [ [-1.27, 2.27, -2.0, 0.83] ]
fb = [ [-1.63, 2.62, -1.72, 0.14, -0.5, -0.98]]
fb=[[-0.75,-0.99,-0.92,-1.68,-0.54,-0.98]] # fb from first 15 years CMIP6 A4X
#fb = [[-1.1051, -0.8333, -1.99907, -1.5852]]
#fb = [ [min(r1) - buffer, max(r2) + buffer, max(r3) + buffer, max(r4) + buffer] ]

###########################
#CO2 FORCING AS A CONSTANT EVERYWHERE
co2 = [8] #w/m2 4~2xCO2, 8~4xCO2, 12~8xCO2

##########################
calib = False #true to enable temp based regional fb, only use with other fb set to 0

if calib: #set other prescribed fb to 0 to allow temp based regional fb to work
    fb=[[0,0,0,0,0,0]]
    co2 = [0] #w/m2 
# fb = [[-4,4,-2,-2]]
##########################
fb_adapt=False #make this true to use feedbacks which change based on local and warm pool temp anoms
if fb_adapt:
    print("FB_ADAPT ON")

# agcm fixes ocean temps at initial and transport at 0
# slab ocean has free temps and transport fixed at equilibrium vals
agcm = False
slab = False

seasonal=False #enable annual cycle of incoming SW for each box
B_param_indiv = False #true to use unique B param for each region
mse_indiv = True #true to use unique mse params for each region


# SET INITIAL TEMPS
# observed ssts
# box1: 302.3 (120-200, 8-8) 302.6 (for 110-160) 302.4(for 120-200)
# box2: 299.4 (200-280, 8-8, zonal grad 2.9) 298.4 (zonal grad of 4.2) 299.09 (for 200-280) (zonal grad of 3.3)
# box3: 294.4 (8-50) 291.6
# box4: 295.7 (-40--8, merid grad 5.9) 294.6 
obst1 = 302.3
obst2 = 299.4
obst3 = 294.4
obst4 = 295.7
obst6 = 276.1 #high lat north
obst7 = 278.4 #high lat south

def tcheck():
    print("T1",np.round(T1-obst1,decimals=2))
    print("T2",np.round(T2-obst2,decimals=2))
    print("T3",np.round(T3-obst3,decimals=2))
    print("T4",np.round(T4-obst4,decimals=2))

#init temps for mean annual CERES SW and tuned mse intercept
### remember to get new control ocean values when retuning ###
# t01=301.57 #302.47 #302.3 #302.238 #302.65 
# t02=298.53 #299.23 #299.5 #299.507 #299.21 
# t03=293.16 #293.94 #293.8 #293.728 #292.28 
# t04=294.62 #295.48 #295.11 #295.013 #294.56 
# t05=293.83 #294.66 #294.328 #293.4 
# t06=276.1
# t07=278.4

#ERA5 mean temps
t01=302.13
t02=299.46
t03=294.12
t04=295.61
t05=294.81 #weighted mean of t03 and t04
t06=276.27
t07=279.11

# initial ocean convergence values
# ot01= -12.018575192140514 #W/m2
# ot02= -35.772093721373444
# ot03= 13.489698008868267
# ot04= 10.660383884555761
# ot05= -6.5e-7
# # initial atmos convergence values
# atm01=-45.20824916692165
# atm02=-39.06954631263214
# atm03=-40.713608945770446
# atm04=-52.755227482917775

################################################################
##### TUNING TO MATCH ERA5 SW + LW & INITIAL TEMPS #####
## use mse int from era5 so all atmos is known, get Ah, Aw values
## based on equations for box 1 & 2, Aw and Ah are required to have different values in each equation
## using the current values of AHT, q term must be ~2.5x bigger in west than east
q=Ah*( np.average([t01,t02],weights=[M1,M2]) - np.average([t03,t04],weights=[M3,M4])) + Aw*(t01-t02)
q3 = q*M3/(M3+M4)
q4 = q-q3

qah = ( np.average([t01,t02],weights=[M1,M2]) - np.average([t03,t04],weights=[M3,M4]))
qaw = (t01-t02)
oc1 = (t02-t01) /M1 * (Cp*rho*h1)
oc2 = (t05-t02) /M2 * (Cp*rho*h2)

msenonwest = np.average( [mse(t02),mse(t03),mse(t04)], weights = [earea,narea,sarea] )
atm_div_w = mse_gamma_w*(mse(t01) - msenonwest) 
mse_int_w = 118 #-(S1*(1-alpha1) - LW1 + atm_div_w + ot01)
atm01 = atm_div_w + mse_int_w

msenoneast = np.average( [mse(t01),mse(t03),mse(t04)], weights = [warea,narea,sarea] )
atm_div_e = mse_gamma_e*(mse(t02) - msenoneast)
mse_int_e = 27 #-(S2*(1-alpha2) - LW2 + atm_div_e + ot02)
atm02 = atm_div_e + mse_int_e

msenonnorth = np.average( [mse(t01),mse(t02),mse(t06)], weights = [warea,earea,hnarea] )
atm_div_n = mse_gamma_n*(mse(t03) - msenonnorth)
mse_int_n = -25 # -(S3*(1-alpha3) - LW3 + atm_div_n + ot03)
atm03 = atm_div_n + mse_int_n

msenonsouth = np.average( [mse(t01),mse(t02),mse(t07)], weights = [warea,earea,hsarea] )
atm_div_s = mse_gamma_s*(mse(t04) - msenonsouth)
mse_int_s = -9 #-(S4*(1-alpha4) - LW4 + atm_div_s + ot04)
atm04 = atm_div_s + mse_int_s

# set up linear system of equations for Ah, Aw and epsilon
# syntax: [X1 * Aw, X2 * Ah] 
# o1 = Aw*qaw*(1-epsilon)*oc1 + Ah*qah*(1-epsilon)*oc1 = -(swlw1 + atm01)
# o2 = Aw*qaw*oc2 + Ah*qah*oc2 = -(swlw2 + atm02)
# o3 = Aw*qaw*(epsilon*(t02-t03)+(1-epsilon)*(t01-t03))/M3 * (Cp*rho*h3)*M3/(M3+M4) + Ah*qah*(epsilon*(t02-t03)+(1-epsilon)*(t01-t03))/M3 * (Cp*rho*h3)*M3/(M3+M4)
# o4 = Aw*qaw*(epsilon*(t02-t04)+(1-epsilon)*(t01-t04))/M4 * (Cp*rho*h4)*M4/(M3+M4) + Ah*qah*(epsilon*(t02-t04)+(1-epsilon)*(t01-t04))/M4 * (Cp*rho*h4)*M4/(M3+M4)
epsilon = 0.48
o1 = [qaw*(1-epsilon)*oc1, qah*(1-epsilon)*oc1, 1]
o2 = [qaw*oc2, qah*oc2, 1]
o3 = [qaw*(epsilon*(t02-t03)+(1-epsilon)*(t01-t03))/M3 * (Cp*rho*h3)*M3/(M3+M4), qah*(epsilon*(t02-t03)+(1-epsilon)*(t01-t03))/M3 * (Cp*rho*h3)*M3/(M3+M4), 1]
o4 = [qaw*(epsilon*(t02-t04)+(1-epsilon)*(t01-t04))/M4 * (Cp*rho*h4)*M4/(M3+M4), qah*(epsilon*(t02-t04)+(1-epsilon)*(t01-t04))/M4 * (Cp*rho*h4)*M4/(M3+M4), 1]
targets = [-(swlw1 + atm01), -(swlw2 + atm02), -(swlw3 + atm03), -(swlw4 + atm04)]
A = [o1,o2,o3,o4]
Aw,Ah,oc_int = np.linalg.lstsq(A,targets)[0]

# then readjust MSE intercept to balance all energy budgets
ot01 = np.dot( o1, [Aw,Ah,oc_int])
ot02 = np.dot( o2, [Aw,Ah,oc_int])
ot03 = np.dot( o3, [Aw,Ah,oc_int])
ot04 = np.dot( o4, [Aw,Ah,oc_int])

mse_int_w = -( swlw1 + atm_div_w + ot01 )
mse_int_e = -( swlw2 + atm_div_e + ot02 )
mse_int_n = -( swlw3 + atm_div_n + ot03 )
mse_int_s = -( swlw4 + atm_div_s + ot04 )

atm_div_hn = mse_gamma_hn*(mse(t06) - mse(t03))
mse_int_hn = -(swlw6 + atm_div_hn)

atm_div_hs = mse_gamma_hs*(mse(t07) - mse(t04)) 
mse_int_hs = -(swlw7 + atm_div_hs)

#####################################################################

zongrad0=t01-t02
norgrad0=(t01+t02)/2 - t03
sougrad0=(t01+t02)/2 - t04

#%% MAIN LOOP
if calib:
    print('CALIBRATION RUN')
#array for temps from all experiments, is appended automatically for each new exp
allT=[]
meanT=[]
allLatent=[]
allR=[]
ocean=[]
atmosdiv=[]
totfb=[]

changenum = 1
for itr in range(changenum): #decide what length you want to run for any parameter suite
    
    # param0 = 14009760591807300
#    delta = param0 * 0.2
    
#    changed = np.arange( param0 - 5*delta, param0 + 5*delta + 0.01, delta)
    # changed = [param0/8, param0, param0*8]

#    Ah = changed[itr]
    # M5 = changed[itr]
    
    
    for c in range(len(co2)): #for each co2 forcing
        
    
        for i in range(len(fb)): #for each feedback set
        
            if fb_adapt:
                print(f"{i+1} of {len(fb)} fb_adapt ON")
            else:
                print(f"{(i+1)*(itr+1)*(c+1) } of {len(fb)*len(co2)*changenum} fb={fb[i]}")
                
            #for sens, write out in between 
            # if i==128:
                # np.save('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.neg2.neg1.1.2.8wm2.meanT.150yr.0-127.npy',np.asarray(meanT))
                # np.save('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.neg2.neg1.1.2.8wm2.allT.150yr.0-127.npy',np.asarray(allT))
                # np.save('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.neg2.neg1.1.2.8wm2.atmdiv.150yr.0-127.npy',np.asarray(atmosdiv))
                # np.save('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.neg2.neg1.1.2.8wm2.oceandiv.150yr.0-127.npy',np.asarray(ocean))
                # np.save('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.neg2.neg1.1.2.8wm2.totfb.150yr.0-127.npy',np.asarray(totfb))
                
                # #after writing out, reset all arrays to free memory
                # allT=[]
                # meanT=[]
                # allLatent=[]
                # allR=[]
                # ocean=[]
                # atmosdiv=[]
                # totfb=[]
                
                # print("files saved and arrays reset")
            
            t1=[t01]
            t2=[t02]
            t3=[t03]
            t4=[t04]
            t5=[t05]
            t6=[t06]
            t7=[t07]
            t1temp=[]
            R1=[]; R2=[]; R3=[]; R4=[]
            H1OLR=[]; H2OLR=[]; H3OLR=[]; H4OLR=[]
            H1Latent=[]; H2Latent=[]; H3Latent=[]; H4Latent=[]
            div=[[],[],[],[],[],[]]
            circ=[[],[],[],[]]
            ot=[[],[],[],[],[]]
            tempfb=[[],[],[],[],[],[]]
        
            Tmean0 = (t1[0]*weights[0] + t2[0]*weights[1] + t3[0]*weights[2] + t4[0]*weights[3])
            if fb_adapt:
                
                fb2param0=fb[i][1]
                fb3param0=fb[i][2]
                fb4param0=fb[i][3]
                fbparams=[[fb2param0],[fb3param0],[fb4param0]]
            for t in range(timesteps-1):
                
                
                #define initial temps
                if t==0: 
                    T1=t01
                    T2=t02
                    T3=t03
                    T4=t04
                    T5=t05
                    T6=t06
                    T7=t07
                    dT1=0 ; dT2=0; dT3=0; dT4=0
                
                    
                #ocean transport m3/s from tropics to extropics
                #q=Ah*( np.average([T1,T2],weights=[M1,M2]) - T3 + Aw*(T1-T2) # for north
                q=Ah*( np.average([T1,T2],weights=[M1,M2]) - np.average([T3,T4],weights=[M3,M4])) + Aw*(T1-T2)  #for north and south
                #q2=Ah*( np.average([T1,T2],weights=[M1,M2]) - T4) + Aw*(T1-T2) #for south
                #weight by volumes of extrop boxes
    #            q = q*M3/(M3+M4)
    #            q2 = q*M4/(M3+M4)
                q4 = q*M4/(M3+M4)
                q3 = q-q4
                
                
                Tmean = 1/np.sum(weights)*(T1*weights[0] + T2*weights[1] + T3*weights[2] + T4*weights[3])
                # mse_mean = 1/np.sum(weights)*(mse(T1)*weights[0] + mse(T2)*weights[1] + mse(T3)*weights[2]
                # + mse(T4)*weights[3])
            ###############################################################
            ## BOX 1 ### WEST #############################################
            ###############################################################
                #fb param in west only depends on local temp
                fb1 = (T1 - t01) * fb[i][0]
                
                #seasonal SW cycle
                if seasonal:
                    S1=SW[0][0][t]
                    
                if not B_param_indiv:
                    Bwest=B
                    Awest=A
                
                if mse_indiv:
                    mse_gamma = mse_gamma_w
                    mse_int = mse_int_w
                    
                # atm_div = mse_gamma*(mse_mean - mse(T1)) + mse_int 
                msenonwest = np.average( [mse(T2),mse(T3),mse(T4)], weights = [earea,narea,sarea] )
                atm_div_w = mse_gamma_w*(mse(T1) - msenonwest) + mse_int_w
                
                #to calibrate equil state
                if calib:
                    R= swlw1 + atm_div_w + co2[c] + fb1 #(Bwest*(T1-273.15) + Awest)
                #once equil T found
                else:
                    R= swlw1 + atm_div_w + co2[c] + fb1 #+ nino[0][0][t]  #use t01 for fixed OLR
                
                if (t+1)%500==0: #write out data every 1/12 of a year
                    tempfb[0].append( fb1 )
                    # sw[0].append(S1*(1-alpha1)); lw[0].append(B*(T1-273.15) + A)
                    div[0].append(atm_div_w)
                    circ[0].append(q*(1-epsilon)*(T2-T1))
                H1= 1/(Cp*rho*h1) * R
                R1.append(R)
            #    H1OLR.append(H1olr)
                
            ###################################################################
            #### BOX 2 ### EAST ############################################### 
            ###################################################################        
                
                if not fb_adapt: #single regional feedback parameter
                    #local feedback: (Tnow - T0) * lambda
                    fb2 = (T2 - t02) * fb[i][1]
                else:
                    # try lambda1 * dt_local + lambda2 * (dt_west - dt_local)
                    fbparam21 = -1.236
                    fbparam22 = -9.846
                    fbparam23 = 8.24
                    fb2 = fbparam21 * (T2 - t02) + fbparam22 * ( (T1-t01) - (T2-t02) ) #+ fbparam23
                
                #seasonal SW cycle
                if seasonal:
                    S2=SW[1][0][t]
                    
                if not B_param_indiv:
                    Beast=B
                    Aeast=A
                    
                if mse_indiv:
                    mse_gamma = mse_gamma_e
                    mse_int = mse_int_e
                
                # atm_div = mse_gamma*(mse_mean - mse(T2)) + mse_int #old version gamma*(Tmean-T2)
                msenoneast = np.average( [mse(T1),mse(T3),mse(T4)], weights = [warea,narea,sarea] )
                atm_div_e = mse_gamma_e*(mse(T2) - msenoneast) + mse_int_e
                
                #to calibrate
                if calib:
                    R= swlw2 + atm_div_e + co2[c] + fb2 #(Beast*(T2-273.15) + Aeast)
                #once equil found
                else:
                    R= swlw2 + atm_div_e + co2[c] + fb2 #+ nino[1][0][t]
                
                if (t+1)%500==0: #write out data every 1/12 of a year
                    tempfb[1].append( fb2 )
                    # sw[1].append(S2*(1-alpha2)); lw[1].append(B*(T2-273.15) + A)
                    div[1].append(atm_div_e)
                    circ[1].append(q*(T4-T2))
                H2= 1/(Cp*rho*h2) * R
                R2.append(R)
            #    H2OLR.append(H2olr)
            
            ########################################################################
            #### BOX 3 ### NORTH ###################################################
            ########################################################################
            
                if not fb_adapt:
                    #local feedback: (Tnow - T0) * lambda
                    fb3 = (T3 - t03) * fb[i][2]
                else:
                    #feedback which becomes more neg with stronger box1 relative warming
                    #effect from west has time delay of four months
                    # feedback param based on cmip6 mean trends. two values and a linear change between them
                    # change happens at a local delta T, as seen in cmip6 data
                    # fbparam31 = -0.67
                    # fbparam32 = -2.0
                    
                    # x1change = 3.3 # local delta T to start change of fb param
                    # x2change = 4.5 # end of change
                    
                    # localdT = T3 - t03
                    
                    # if localdT < x1change:
                    #     fbparam3 = fbparam31
                    # elif localdT > x2change:
                    #     fbparam3 = fbparam32
                    # else:
                    #     fbparam3 = linchange( fbparam31, fbparam32, x1change, x2change, localdT)
                    
                    # fb3 = (T3 - t03) * fbparam3
                    
                    # try lambda1 * dt_local + lambda2 * (dt_west - dt_local)
                    fbparam31 = -1.188
                    fbparam32 = -3.897
                    fbparam33 = 7.083
                    fb3 = fbparam31 * (T3 - t03) + fbparam32 * ((T1-t01) - (T3-t03)) #+ fbparam33
                
                #seasonal SW cycle
                if seasonal:
                    S3=SW[2][0][t]
                
                if not B_param_indiv:
                    Bnorth=B
                    Anorth=A
                if mse_indiv:
                    mse_gamma = mse_gamma_n
                    mse_int = mse_int_n
                
                # atm_div = mse_gamma*(mse_mean - mse(T3)) + mse_int 
                msenonnorth = np.average( [mse(T1),mse(T2),mse(T6)], weights = [warea,earea,hnarea] )
                atm_div_n = mse_gamma_n*(mse(T3) - msenonnorth) + mse_int_n
                
                #to calibrate
                if calib:
                    R= swlw3 + atm_div_n + co2[c] + fb3 # (Bnorth*(T3-273.15) + Anorth)
                #once equil found
                else:
                    R= swlw3 + atm_div_n + co2[c] + fb3 #+ nino[2][0][t]
                
                if (t+1)%500==0: #write out data every 1/12 of a year
                    tempfb[2].append( fb3 )
                    # sw[2].append(S3*(1-alpha3)); lw[2].append(B*(T3-273.15) + A)
                    div[2].append(atm_div_n)
                    circ[2].append(q*epsilon*(T2-T3) + q*(1-epsilon)*(T1-T3))
                H3= 1/(Cp*rho*h3) * R
                R3.append(R)
            #    H3OLR.append(H3olr)
            #    H3Latent.append(H3latent)
            
            ########################################################################
            #### BOX 4 ### SOUTH ###################################################
            ########################################################################
            
                if not fb_adapt:
                    #local feedback: (Tnow - T0) * lambda
                    fb4 = (T4 - t04) * fb[i][3]
                else:
                    # try lambda1 * dt_local + lambda2 * (dt_west - dt_local)
                    fbparam41 = -0.557
                    fbparam42 = -4.725
                    fbparam43 = 7.877
                    fb4 = fbparam41 * (T4 - t04) + fbparam42 * ((T1-t01) - (T4-t04)) #+ fbparam43
                
                #seasonal SW cycle
                if seasonal:
                    S4=SW[3][0][t]
                
                if not B_param_indiv:
                    Bsouth=B
                    Asouth=A
                    
                if mse_indiv:
                    mse_gamma = mse_gamma_s
                    mse_int = mse_int_s
                
                # atm_div = mse_gamma*(mse_mean - mse(T4)) + mse_int 
                msenonsouth = np.average( [mse(T1),mse(T2),mse(T7)], weights = [warea,earea,hsarea] )
                atm_div_s = mse_gamma_s*(mse(T4) - msenonsouth) + mse_int_s

                
                #to calibrate
                if calib:
                    R= swlw4 + atm_div_s + co2[c] + fb4 # (Bsouth*(T4-273.15) + Asouth)
                #once equil found
                else:
                    R= swlw4 + atm_div_s + co2[c] + fb4 #+ aaht #+ nino[2][0][t]
                
                if (t+1)%500==0: #write out data every 1/12 of a year
                    tempfb[3].append( fb4 )
                    # sw[3].append(S3*(1-alpha3)); lw[2].append(B*(T3-273.15) + A)
                    div[3].append(atm_div_s)
                    circ[3].append(q*epsilon*(T2-T3) + q*(1-epsilon)*(T1-T3))
                H4= 1/(Cp*rho*h4) * R
                R4.append(R)
            #    H3OLR.append(H3olr)
            #    H3Latent.append(H3latent)
            
            
            ########################################################################
            #### BOX 6 ### HIGH LATITUDE NORTH #####################################
            ########################################################################
            
                
                fb6 = (T6 - t06) * fb[i][4]
                
                
                if not B_param_indiv:
                    Bhnorth=B
                    Ahnorth=A
                    
                if mse_indiv:
                    mse_gamma = mse_gamma_hn
                    mse_int = mse_int_hn
    
                atm_div_hn = mse_gamma_hn*(mse(T6) - mse(T3)) + mse_int_hn
                
                #to calibrate
                if calib:
                    R= swlw6 + atm_div_hn + co2[c] + fb6
                #once equil found
                else:
                    R= swlw6 + atm_div_hn + co2[c] + fb6 
                
                if (t+1)%500==0: #write out data every 1/12 of a year
                    tempfb[4].append( fb6 )
                # sw[3].append(S3*(1-alpha3)); lw[2].append(B*(T3-273.15) + A)
                    div[4].append(atm_div_hn)
                # circ[3].append(q*epsilon*(T2-T3) + q*(1-epsilon)*(T1-T3))
                H6= 1/(Cp*rho*h6) * R
                # R4.append(R)
            #    H3OLR.append(H3olr)
            #    H3Latent.append(H3latent)
            
            ########################################################################
            #### BOX 7 ### HIGH LATITUDE SOUTH #####################################
            ########################################################################
            
                
                fb7 = (T7 - t07) * fb[i][5]
                
                
                if not B_param_indiv:
                    Bhsouth=B
                    Ahsouth=A
                    
                if mse_indiv:
                    mse_gamma = mse_gamma_hs
                    mse_int = mse_int_hs
    
                atm_div_hs = mse_gamma_hs*(mse(T7) - mse(T4)) + mse_int_hs
                
                #to calibrate
                if calib:
                    R= swlw7 + atm_div_hs + co2[c] + fb7
                #once equil found
                else:
                    R= swlw7 + atm_div_hs + co2[c] + fb7 
                
                if (t+1)%500==0: #write out data every 1/12 of a year
                    tempfb[5].append( fb7 )
                
                    div[5].append(atm_div_hs)
                # circ[3].append(q*epsilon*(T2-T3) + q*(1-epsilon)*(T1-T3))
                H7= 1/(Cp*rho*h7) * R
                # R4.append(R)
            #    H3OLR.append(H3olr)
            #    H3Latent.append(H3latent)
            
            ########################################################################
                
                
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
                    #ocean transport, units T*m3/s
                    
                    # east feeding west with water not lost to ekman
                    ocean1 = q*(1-epsilon)*(T2-T1) + oc_int*M1/(Cp*rho*h1) #convert intercept W/m2 -> T*m3/s
                    #undercurrent (T5) feeding into east (T2)
                    ocean2 = q*(T5-T2) + oc_int*M2/(Cp*rho*h2)
                    #east and west feeding north via ekman
                    ocean3 = q3*epsilon*(T2-T3) + q3*(1-epsilon)*(T1-T3) + oc_int*M3/(Cp*rho*h3)
                    #east and west feeding south via ekman
                    ocean4 = q4*epsilon*(T2-T4) + q4*(1-epsilon)*(T1-T4) + oc_int*M4/(Cp*rho*h4)
                    #undercurrent being fed by north and south, as average of two weighted by volumes of each
                    ocean5 = q3*(T3-T5) + q4*(T4-T5) #np.average( [q*(T3-T5), q2*(T4-T5)], weights=[M3,M4] ) 
                    
                    #temperature evolution equations
                    T1=T1 + dt/M1 * (M1*H1 + ocean1) 
                    T2=T2 + dt/M2 * (M2*H2 + ocean2) #+ nino[1][0][t] * dt/(3600*24*30)
                    T3=T3 + dt/M3 * (M3*H3 + ocean3) #+ (-0.04*dt/(3600*24*30)) #add cooling trend from upwelling of .02 deg/month
                    T4=T4 + dt/M4 * (M4*H4 + ocean4) #+ (-0.04*dt/(3600*24*30))
                    T5=T5 + dt/M5 * ocean5 
                    T6=T6 + dt*H6 # High lat NH, assuming no ocean connection
                    T7=T7 + dt*H7 # High lat SH, assuming no ocean connection
                    
                    #get ocean heat transport in W/m2
                    o1wm2 = ocean1/M1 * (Cp*rho*h1)
                    o2wm2 = ocean2/M2 * (Cp*rho*h2)
                    o3wm2 = ocean3/M3 * (Cp*rho*h3)
                    o4wm2 = ocean4/M4 * (Cp*rho*h4)
                    o5wm2 = ocean5/M5 * (Cp*rho*underdepth)
                
                dT1=T1-t1[0]
                dT2=T2-t2[0]
                dT3=T3-t3[0]
                
                if fb_adapt:
                    t1temp.append(T1) #every timestep of T1 to calc adaptive fb
                
                if (t+1)%500==0: #write out data every 1/12 of a year
                    t1.append(T1)
                    t2.append(T2)
                    t3.append(T3)
                    t4.append(T4)
                    t5.append(T5)
                    t6.append(T6)
                    t7.append(T7)
                    
                    ot[0].append(o1wm2)
                    ot[1].append(o2wm2)
                    ot[2].append(o3wm2)
                    ot[3].append(o4wm2)
                    ot[4].append(o5wm2)
            
            t1=np.asarray(t1)
            t2=np.asarray(t2)
            t3=np.asarray(t3)
            t4=np.asarray(t4)
            t5=np.asarray(t5)
            t6=np.asarray(t6)
            t7=np.asarray(t7)
            
#            if itr !=2: #for runs of different lengths, add nan to previous runs to equalize lengths
#                t1 = np.append(t1, np.zeros(len(t1)) + np.nan)
#                t2 = np.append(t2, np.zeros(len(t2)) + np.nan)
#                t3 = np.append(t3, np.zeros(len(t3)) + np.nan)
#                t4 = np.append(t4, np.zeros(len(t4)) + np.nan)
#                t5 = np.append(t5, np.zeros(len(t5)) + np.nan)
                
            ocean.append( np.asarray(ot) )
            totfb.append( np.array(tempfb) )
            allT.append([t1,t2,t3,t4,t5,t6,t7])
            allR.append([R1,R2,R3,R4])
            atmosdiv.append(np.asarray(div))
            
        #####################################

            Tmean = (t1*weights[0] + t2*weights[1] + t3*weights[2]+ t4*weights[3])
            Tmean0 = (t1[0]*weights[0] + t2[0]*weights[1] + t3[0]*weights[2] + t4[0]*weights[3])
            Tmeandiff = Tmean - Tmean0
            
            meanT.append(np.round(Tmeandiff,decimals=2))
    

if calib:
    tcheck()
    print(f"t01={T1}")
    print(f"t02={T2}")
    print(f"t03={T3}")
    print(f"t04={T4}")
    print(f"t05={T5}")
    print(f"t06={T6}")
    print(f"t07={T7}")
    print(f"atm01={atmosdiv[0][0][-1]}")
    print(f"atm02={atmosdiv[0][1][-1]}")
    print(f"atm03={atmosdiv[0][2][-1]}")
    print(f"atm04={atmosdiv[0][3][-1]}")
    print(f"atm06={atmosdiv[0][4][-1]}")
    print(f"atm07={atmosdiv[0][5][-1]}")
    print(f"ot01={ocean[0][0][-1]}")
    print(f"ot02={ocean[0][1][-1]}")
    print(f"ot03={ocean[0][2][-1]}")
    print(f"ot04={ocean[0][3][-1]}")
    print(f"ot05={ocean[0][4][-1]}")
#%% write out data
    
# np.save('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.neg2.neg1.1.2.8wm2.meanT.150yr.128-256.npy',np.asarray(meanT))
# np.save('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.neg2.neg1.1.2.8wm2.allT.150yr.128-256.npy',np.asarray(allT))
# np.save('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.neg2.neg1.1.2.8wm2.atmdiv.150yr.128-256.npy',np.asarray(atmosdiv))
# np.save('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.neg2.neg1.1.2.8wm2.oceandiv.150yr.128-256.npy',np.asarray(ocean))
# np.save('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.neg2.neg1.1.2.8wm2.totfb.150yr.128-256.npy',np.asarray(totfb))

#%% #########################################################################################
################ PLOTS - EVERYTHING BELOW HERE ARE VARIOUS PLOTS ############################
#############################################################################################

exps=[str(fb[i]) for i in range(len(fb))]
cmap = plt.get_cmap('jet')
expcols=cmap(np.linspace(0,1,len(fb)))#
expcols=['tab:blue','tab:orange','tab:green','tab:purple']

plt.figure(0, figsize=(10,5))
expnum=len(allT)
plt.subplot(1,3,1)
for i in range(expnum):
    plt.plot(allT[i][0]-allT[i][1],color=expcols[i]) # T1 - T2
plt.title('Eq Gradient T1-T2')
plt.ylabel('T1-T2')
plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Years') 
plt.legend()

plt.subplot(1,3,2)
for i in range(expnum):
    plt.plot( (allT[i][0]+allT[i][1])/2 - np.average([allT[i][2],allT[i][3] ],weights=[M3,M4],axis=0) ,color=expcols[i]) # T1 - T2
plt.title('Merid Gradient (T1+T2)/2 - (T3+T4)/2')
plt.ylabel('Merid')
plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Years') 
plt.legend()

plt.subplot(1,3,3)
for i in range(expnum):
    plt.plot( allT[i][0]-allT[i][1], (allT[i][0]+allT[i][1])/2 - np.average([allT[i][2],allT[i][3] ],weights=[M3,M4],axis=0) ,color=expcols[i]) # T1 - T2
plt.title('Merid vs Zonal')
plt.ylabel('Merid')
#plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Zonal') 
plt.legend()
#plt.legend(['6e15','8e15','12e15'])

############
plt.figure(1)
plt.title("Temperatures")
expnum=len(allT)
for i in range(expnum):
    plt.plot(allT[i][0] - t01,color=expcols[i],label=co2[0])
    plt.plot(allT[i][1] - t02,color=expcols[i])
    plt.plot(allT[i][2] - t03,color=expcols[i])
    plt.plot(allT[i][3] - t04,color=expcols[i])
    plt.plot(allT[i][4] - t05,color=expcols[i])

plt.annotate("Trop W",xy=(0,allT[0][0][0]-0.3))
plt.annotate("Trop E",xy=(0,allT[0][1][0]-0.3))
pltlen=int(len(allT[0][0])-0.2*len(allT[0][0]))
plt.annotate("Ex Trop N",xy=(pltlen,allT[0][2][pltlen]))
plt.annotate("Ex Trop S",xy=(pltlen,allT[0][3][pltlen]))
plt.annotate("UnderCurrent",xy=(0,allT[0][4][100]))

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
#    plt.plot(allR[i][3],color=expcols[i])
#    
#plt.annotate("Trop W",xy=(0,allR[0][0][0]-0.3))
#plt.annotate("Trop E",xy=(0,allR[0][1][0]-0.3))
#plt.annotate("Ex Trop N",xy=(0,allR[0][2][0]))
#plt.annotate("Ex Trop S",xy=(0,allR[0][3][0]))
#
#plt.legend()
#
#############
plt.figure(3)
plt.title("Mean Temp Change")
expnum=len(meanT)
for i in range(expnum):
    plt.plot(meanT[i],color=expcols[i],label=co2[0])
    
plt.ylabel('K')    
plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Years') 
plt.legend()
#
#############
##only works for last exp right now
plt.figure(4)
plt.title('Atm Div - Relative warming = more negative')
plt.xlabel('Years') 
expnum=len(atmosdiv)
for i in range(expnum):
    plt.plot(atmosdiv[i][0][:],label='West')
    plt.plot(atmosdiv[i][1][:],label='East')
    plt.plot(atmosdiv[i][2][:],label='North')
    plt.plot(atmosdiv[i][3][:],label='South')
    
# plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
# plt.annotate("Trop W",xy=(0,div[0][0]-0.3))
# plt.annotate("Trop E",xy=(0,div[1][0]-0.3))
# plt.annotate("Ex Trop N",xy=(0,div[2][10000]))
# plt.annotate("Ex Trop S",xy=(0,div[3][10000]))

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

plt.figure(6,figsize=(7,6))
if fb_adapt:
    plt.title("λ=coupled",weight='bold',fontsize=13)
else:
    plt.title(f"λ={fb[0]}",weight='bold',fontsize=13)
expnum=len(ocean)
for i in range(expnum):
    plt.plot(ocean[i][0],label='West')
    plt.plot(ocean[i][1],label='East')
    plt.plot(ocean[i][2],label='North')
    plt.plot(ocean[i][3],label='South')
    plt.plot(ocean[i][4],label='Undercurrent')

# plt.annotate("Trop W",xy=(0,ocean[0][0][0]-0.3))
# plt.annotate("Trop E",xy=(0,ocean[0][1][0]-0.3))
# pltlen=int(len(ocean[0][0])-0.2*len(ocean[0][0]))
# plt.annotate("Ex Trop N",xy=(pltlen,ocean[0][2][pltlen]))
# plt.annotate("Ex Trop S",xy=(pltlen,ocean[0][3][pltlen]))
# plt.annotate("UnderCurrent",xy=(0,ocean[0][4][pltlen]))

plt.hlines(y=0,xmin=0,xmax=6,linestyle='--',color='grey',alpha=0.5)
plt.yticks(fontsize=13)
plt.ylabel('OHT (W/m^2)', fontsize=15)
plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round(),fontsize=12)
plt.xlabel('Years',fontsize=15) 
plt.legend()


#only for one exp
# plt.figure(7)
# plt.title("Ocean heat transport vs local T")
# expnum=len(ocean)
# for i in range(1):
#     plt.plot(allT[i][0][1:] - t01, ocean[i][0],label='West')
#     plt.plot(allT[i][1][1:] - t02, ocean[i][1],label='East')
#     plt.plot(allT[i][2][1:] - t03, ocean[i][2],label='North')
#     plt.plot(allT[i][3][1:] - t04, ocean[i][3],label='South')
#     plt.plot(allT[i][4][1:] - t05, ocean[i][4],label='Undercurrent')

# # plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
# plt.hlines(y=0,xmin=0,xmax=6,linestyle='--',color='grey',alpha=0.5)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.ylabel('OHT (W/m^2)', fontsize=15)
# plt.xlabel('ΔT (K)', fontsize=15) 
# plt.legend()

#%% solve for lambda at each timestep
# energy budget plot: (Cp*rho*h) * dT/dt = lambda*dT - atmdiv - ocndiv

#conversions from K/s to W/m2
convfactor = [(Cp*rho*h1), (Cp*rho*h2), (Cp*rho*h3), (Cp*rho*h4)]
T0=[t01,t02,t03,t04,t05]
swlw =[S1*(1-alpha1) - LW1, S2*(1-alpha2) - LW2,
       S3*(1-alpha3) - LW3, S4*(1-alpha4) - LW4]
labels = ['West','East','North','South']
lamdas=[]
colors=['tab:blue','tab:orange','tab:green','tab:purple']

#lambda plot: use to solve for lambda if not fixed
for i in range(4):
    dT = (allT[0][i][1:] - allT[0][i][:-1]) * convfactor[i]
    dTdt = dT/(dt*(timesteps/len(allT[0][0]))) #change in T per timestep, accounts for not writing out every time step
    
    deltaT = allT[0][i][:-1] - T0[i] #anomalous T at previous timestep
    
    lamda = (dTdt  - swlw[i] - co2[0] - atmosdiv[0][i] - ocean[0][i])/deltaT
    lamdas.append(lamda)
    
    plt.plot(lamda,label=labels[i],color=colors[i])
    
plt.xticks(ticks=np.linspace(0,len(lamda),6),labels=np.linspace(0,int(years),6).astype(int),fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=15)
plt.xlabel('Years',fontsize=15)
plt.ylabel('λ',fontsize=15)
plt.hlines(0, xmin=0, xmax=len(lamda),linestyle='--',color='grey',alpha=0.6)

#%% plot fractional contribution of each term to dT/dt

for i in range(4):
    plt.subplot(2,2,i+1)
    
    dT = (allT[0][i][1:] - allT[0][i][:-1]) * convfactor[i]
    dTdt = (dT/(dt*(timesteps/len(allT[0][0])))) #change in T per timestep
    
    deltaT = allT[0][i][:-1] - T0[i] #anomalous T at previous timestep
    
    # add abs of all values to get total energy contributions
    total = swlw[i] + co2[0] + np.abs(atmosdiv[0][i]) + np.abs(ocean[0][i]) + np.abs(lamdas[i]*deltaT)
    
    #filled plot for each term
    fbdiff = 1-( np.abs(lamdas[i]*deltaT) / total )
    plt.fill_between( np.arange(len(total)), 1, fbdiff , color=colors[0])
    
    ocndiff = fbdiff - np.abs(ocean[0][i]) / total
    plt.fill_between( np.arange(len(total)), fbdiff, ocndiff , color=colors[1])
    
    atmdiff = ocndiff - np.abs(atmosdiv[0][i]) / total
    plt.fill_between( np.arange(len(total)), ocndiff, atmdiff, color=colors[2] )
    
    const = atmdiff - (swlw[i] + co2[0])/total
    plt.fill_between( np.arange(len(total)), atmdiff, const , color=colors[3])
    
    plt.xticks(ticks=np.linspace(0,len(total),6),labels=np.linspace(0,int(years),6).astype(int))
    if i>1:
        plt.xlabel('Years') 
    plt.title(labels[i],fontsize=15)

#%% plot fractional contribution of each term to lambda
titles=['West','East','North','South']
convfactor = [(Cp*rho*h1), (Cp*rho*h2), (Cp*rho*h3), (Cp*rho*h4)]
swlw =[S1*(1-alpha1) - LW1, S2*(1-alpha2) - LW2,
       S3*(1-alpha3) - LW3, S4*(1-alpha4) - LW4]
T0=[t01,t02,t03,t04,t05]
A0=[atm01,atm02,atm03,atm04]
O0=[ot01,ot02,ot03,ot04]
colors=['tab:blue','tab:orange','tab:green','tab:purple']
exp=0
for i in range(3):
    plt.subplot(2,2,i+1)
    
    
    dT = (allT[exp][i][1:] - allT[exp][i][:-1]) * convfactor[i]
    dTdt = (dT/(dt*(timesteps/len(allT[0][0])))) #change in T per timestep
    
    deltaT = allT[exp][i][:-1] - T0[i] #anomalous T at previous timestep
    deltaA = atmosdiv[exp][i] - A0[i]
    deltaO = ocean[exp][i] - O0[i]
    
    # add abs of all values to get total energy contributions
    total =  ( dTdt + co2[0] + np.abs(deltaA) + np.abs(deltaO) ) #(1/deltaT) *
    
    #filled plot for each term
    
    dTdtdiff = 1-( np.abs(dTdt) / total )
    plt.fill_between( np.arange(len(total)), 1, dTdtdiff , color=colors[0],label="dT/dt")
    
    ocndiff = dTdtdiff - np.abs(deltaO) / total
    plt.fill_between( np.arange(len(total)), dTdtdiff, ocndiff , color=colors[1], label = "OHT")
    if np.mean(deltaO)<0:
        plt.fill_between( np.arange(len(total)), dTdtdiff, ocndiff ,color='none',edgecolor='black',hatch='X')
    
    atmdiff = ocndiff - np.abs(deltaA) / total
    plt.fill_between( np.arange(len(total)), ocndiff, atmdiff, color=colors[2], label = "AHT" )
    if np.mean(deltaA)<0:
        plt.fill_between( np.arange(len(total)), ocndiff, atmdiff,color='none',edgecolor='black',hatch='X' )

    toadiff = atmdiff - (co2[0])/total
    plt.fill_between( np.arange(len(total)), atmdiff, toadiff, color=colors[3], label="CO2 Forcing" )
    
    plt.xticks(ticks=np.linspace(0,len(total),6),labels=np.linspace(0,int(years),6).astype(int))
    if i>1:
        plt.xlabel('Years',fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(titles[i],fontsize=15,weight='bold')
    plt.legend()
    
#%% plot total contribution of each term to lambda

titles=['West','East','North','South']

convfactor = [(Cp*rho*h1), (Cp*rho*h2), (Cp*rho*h3), (Cp*rho*h4)]
T0=[t01,t02,t03,t04,t05]
A0=[atm01,atm02,atm03,atm04]
O0=[ot01,ot02,ot03,ot04]
colors=['tab:blue','tab:orange','tab:green','tab:purple']
exp=0
for i in range(4):
    plt.subplot(2,2,i+1)
    
    
    dT = (allT[exp][i][1:] - allT[exp][i][:-1]) * convfactor[i]
    dTdt = (dT/(dt*(timesteps/len(allT[0][0])))) #change in T per timestep
    
    deltaT = allT[exp][i][:-1] - T0[i] #anomalous T at previous timestep
    deltaA = atmosdiv[exp][i] - A0[i]
    deltaO = ocean[exp][i] - O0[i]
    
    #filled plot for each term
    
    dTdtpos = np.where(dTdt>0,dTdt,0)
    dTdtneg = np.where(dTdt<0,dTdt,0) 
    plt.fill_between( np.arange(len(dTdt)), 0, dTdtpos , color=colors[0],label="dT/dt",alpha=0.8)
    plt.fill_between( np.arange(len(dTdt)), 0, dTdtneg , color=colors[0],alpha=0.8)
    
    deltaOpos = np.where(deltaO>0,deltaO,0)
    deltaOneg = np.where(deltaO<0,deltaO,0)
    plt.fill_between( np.arange(len(dTdt)), dTdtpos, np.nansum([dTdtpos,deltaOpos],axis=0) , color=colors[1], label = "OHT",alpha=0.8)
    plt.fill_between( np.arange(len(dTdt)), dTdtneg, np.nansum([dTdtneg,deltaOneg],axis=0) , color=colors[1],alpha=0.8)
    
    deltaApos = np.where(deltaA>0,deltaA,0)
    deltaAneg = np.where(deltaA<0,deltaA,0)
    plt.fill_between( np.arange(len(dTdt)), np.nansum([dTdtpos,deltaOpos],axis=0), np.nansum([dTdtpos,deltaOpos,deltaApos],axis=0) , color=colors[2], label = "AHT",alpha=0.8)
    plt.fill_between( np.arange(len(dTdt)), np.nansum([dTdtneg,deltaOneg],axis=0), np.nansum([dTdtneg,deltaOneg,deltaAneg],axis=0) , color=colors[2],alpha=0.8)
    
    plt.fill_between( np.arange(len(dTdt)), np.nansum([dTdtpos,deltaOpos,deltaApos],axis=0), np.nansum([dTdtpos,deltaOpos,deltaApos],axis=0)+co2[0], color=colors[3], label="CO2 Forcing",alpha=0.8 )
    
    plt.hlines( y=0 ,xmin=0,xmax=len(dTdt), linestyle='--', color='silver',alpha=1)
    plt.plot( (dTdt+deltaA+deltaO+co2[0]), linewidth=2,color='black',label='Total')
    plt.plot( (dTdt+deltaA+deltaO+co2[0])+totfb[0][i], linewidth=2,color='gold',label='Total + FB')
    plt.xticks(ticks=np.linspace(0,len(dTdt),6),labels=np.linspace(0,int(years),6).astype(int))
    if i>1:
        plt.xlabel('Years',fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(titles[i],fontsize=15,weight='bold')
    plt.legend(fontsize=14)
#%% gregory plots


vols = [M1,M2,M3,M4]
#ΔTOA = F - λΔT, as SW & LW are fixed
#regional plots
titles=['West','East','North','South']
for i in range(4):
    plt.subplot(2,2,i+1)
    
    deltaTOA = co2[0] + totfb[0][i]
    
    deltaT = allT[0][i][1:] - allT[0][i][0]
    
    y = deltaTOA # {::500] take monthly (every 500) to make plotting easier
    x = deltaT
    plt.scatter( x, y , alpha=0.5, s=40, facecolor = 'white', edgecolors='grey')
    
    #first X years
    timecut=15*12 #years * months 
    slope, inter = np.polyfit( x[:timecut], y[:timecut], 1)
    plt.plot( x[:timecut], x[:timecut]*slope + inter, label = f"0-{int(timecut/12)} slope={np.round(slope,decimals=2)}", linewidth = 2)
    #remaining years
    slope, inter = np.polyfit( x[timecut:], y[timecut:], 1)
    plt.plot( x[timecut:], x[timecut:]*slope + inter, label = f"{int(timecut/12)}-150 slope={np.round(slope,decimals=2)}", linewidth = 2)
    
    if i>1:
        plt.xlabel('ΔT (K)',fontsize=15)
    plt.ylabel('ΔTOA (W/m^2)',fontsize=15)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(titles[i],fontsize=14,weight='bold')
    
#%% "global" mean gregory plot
x = meanT[0][1:][::500] #weighted mean anomalous T
y = weights[0]*(co2[0] + totfb[0][0]) + weights[1]*(co2[0] + totfb[0][1]) + weights[2]*(co2[0] + totfb[0][2]) + weights[3]*(co2[0] + totfb[0][3])
y = y[::500] #take value every month to reduce plot time
plt.scatter( x, y , alpha=0.5, s=40, facecolor = 'white', edgecolors='grey')

#first X years
timecut=15*12 #years * months 
slope, inter = np.polyfit( x[:timecut], y[:timecut], 1)
plt.plot( x[:timecut], x[:timecut]*slope + inter, label = f"0-{int(timecut/12)} slope={np.round(slope,decimals=2)}", linewidth = 2)
#remaining years
slope, inter = np.polyfit( x[timecut:], y[timecut:], 1)
plt.plot( x[timecut:], x[timecut:]*slope + inter, label = f"{int(timecut/12)}-150 slope={np.round(slope,decimals=2)}", linewidth = 2)

plt.xlabel('ΔT',fontsize=15)
plt.ylabel('ΔTOA',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title("Mean of all 4 regions", weight='bold')
plt.legend(fontsize=14)
#%%
# period = int((timesteps-1)/len(deltaT))
# deltat = np.arange( 0, timesteps, period,dtype=np.float32)[1:]*dt
# deltatmon = np.arange( 0, timesteps, 500,dtype=np.float32)[1:]*dt
# ΔH = volume * ΔT/Δt -> convert to W/m2 via convfactor
# deltaT_time = allT[0][box][1:] - allT[0][box][:-1]
# deltaT_timemon = deltaT_time[::500][:-1]
# deltaH = vols[box] * deltaT_time / dt * convfactor[box]
# deltaHmon = vols[box] * deltaT_timemon / (dt*500) * convfactor[box]

# deltaatmdiv = (np.asarray(div[box][::period][1:]) - div[box][0])
# atmdiv = div[box][::period][1:]
# atmdivmon = atmdiv[::500][:-1]

# if len(ocean[0][box]) == 1799:
#     ocndiv = np.append(ocean[0][box], ocean[0][box][-1])
    
# deltaocndiv = (ocndiv[1:] - ocean[0][box][0]) * convfactor[box]
# ocndiv = ocean[0][box][1:] * convfactor[box]
# ocndivmon = ocndiv[::500][:-1]

# F + deltaT*lambda (plus because lambda is negative)



# #now subtract atm div as well
# y = y - deltaatmdiv
# plt.scatter( x, y , alpha=0.5, s=40, facecolor = 'white', edgecolors='tab:orange')
# slope, inter = np.polyfit( x, y, 1)
# plt.plot( x, x*slope + inter, label = f"F - λΔT - Δatm_div slope={np.round(slope,decimals=2)}", linewidth = 2)

# #now subtract ocean div as well
    
# y = y - deltaocndiv
# plt.scatter( x, y , alpha=0.5, s=40, facecolor = 'white', edgecolors='tab:green')
# slope, inter = np.polyfit( x, y, 1)
# plt.plot( x, x*slope + inter, label = f"F - λΔT - Δatm_div - Δocean slope={np.round(slope,decimals=2)}", linewidth = 2)

# λ = ΔH - F - Δatm_div - Δocean
# y = deltaH - co2[0] - deltaatmdiv - deltaocndiv
# y = deltaH  - atmdiv - ocndiv - swlw[box]
# y2 = deltaH - co2[0] - atmdiv - ocndiv - swlw[box]
# plt.scatter( x, y , alpha=0.5, s=40, facecolor = 'white', edgecolors='tab:blue')
# slope, inter = np.polyfit( x[:], y[:], 1)
# plt.plot( x, x*slope + inter, label = f"ΔH - atm_div - ocean - SW + LW slope={np.round(slope,decimals=2)}", linewidth = 2)

# plt.scatter( x, y2 , alpha=0.5, s=40, facecolor = 'white', edgecolors='tab:red')
# slope, inter = np.polyfit( x[:], y2[:], 1)
# plt.plot( x, x*slope + inter, label = f"ΔH - F - atm_div - ocean - SW + LW slope={np.round(slope,decimals=2)}", linewidth = 2)

# plt.legend()
# plt.title(f"Box {box+1} λ = {np.round(fb[0][box], decimals=2)}")
# plt.xlabel("ΔT Local")
# plt.ylabel("W/m2")

#%% energy imbalances?
plt.plot(deltaH, label='ΔH = volume * ΔT/Δt')
plt.plot(deltaT*fb[0][box] + co2[0] + deltaatmdiv +deltaocndiv, label = 'F + λΔT + Δatm_div + Δocean slope')
plt.xlabel('Time')
plt.ylabel('W/m2')
plt.title(f"Box {box+1}")
plt.legend()

#%% calculate lambda explicitly
l_est=[]
l_estmon=[]
for i in range(len(deltaT)):
    l_est.append(( Cp*rho*h1 * deltaT_time[i]/dt - ocndiv[i] - swlw[box] - atmdiv[i] - co2[0] ) / deltaT[i] )
    
for i in range(len(deltaTmon)):
    l_estmon.append(( Cp*rho*h1 * deltaT_timemon[i]/deltatmon[i] - ocndivmon[i] - swlw[box] - atmdivmon[i] - co2[0] ) / deltaTmon[i] )
    
plt.plot(l_est,alpha=0.8)
# plt.plot(l_estmon,alpha=0.8,linestyle=':')
# plt.plot(np.asarray(l_est[::500][:-1]) - np.asarray(l_estmon))
plt.hlines(fb[0][box], xmin=0,xmax=900000, linestyle='--', color='grey',alpha=0.9)
plt.xlabel('timestep')
plt.ylabel('predicted λ')
# plt.ylim([-5,0.5])
plt.title(f"Box {box+1} λ = {np.round(fb[0][box], decimals=4)}")

#or print the mean value over a time range
time1=750000
time2=900000
    
l_est_mean = ( Cp*rho*h1 * np.mean(deltaT_time[time1:time2])/dt - 
              np.mean(ocndiv[time1:time2]) - swlw[box] - np.mean(atmdiv[time1:time2]) - co2[0] ) / np.mean(deltaT[time1:time2])
print(l_est_mean)
#%% OPEN AND PLOT SENSITIVITY RUNS

meanT=[]
#meanT.append( np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/cmip6.kernelfb.8wm2.meanT.150.npy') )
#meanT.append( np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.cmipintervals.8wm2.meanT.150.npy') )
#meanT.append( np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.undervol.8wm2.meanT.600.npy') )
# meanT.append( np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.neg4to1.agu.8wm2.meanT.150.npy') )
#meanT.append( np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/fb.by1.16wm2.meanT.npy') )
allT=[]
#allT.append( np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/cmip6.kernelfb.8wm2.allT.150.npy') )
#allT.append( np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.cmipintervals.8wm2.allT.150.npy') )
#allT.append( np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.undervol.8wm2.allT.600.npy') )
allT1= [ np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.neg2.neg1.1.2.8wm2.allT.150yr.0-127.npy') ]
allT2= [ np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/sens.neg2.neg1.1.2.8wm2.allT.150yr.128-256.npy') ]
#allT.append( np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/fb.by1.16wm2.allT.npy') )
allT = np.concatenate( (allT1[0], allT2[0] ) )
allT = allT[:,:,::500] #take only monthly values for easy plotting
#meanTann = np.asarray( [annualmean(meanT[i,:]) for i in range(len(meanT))] )
#allTann = np.zeros(shape=(allT.shape[0],4,int(years)))
#for box in range(4):
#    allTann[:,box,:] = np.asarray( [annualmean(allT[i,box,:]) for i in range(len(allT))] )
#    

allTmon = []
for i in range(len(allT)):
#    if i==0:
    allTmon.append(allT[i])
#    else:
#        allTmon.append( np.zeros(shape=(allT[i].shape[0],4,int(50*12))) )
#        for box in range(4):
#            allTmon[i][:,box,:] = np.asarray( [monmean(allT[i][j,box,:]) for j in range(len(allT[i]))] )

meanTmon=[]
for i in range(len(meanT)):
#    if i==0: #for new longer files
    #had a bug in meanT calc after adding 5th box, manually recalc
#    meanTmon.append([])
#    for j in range(len(meanT[i])):
#        newmean = calcmean(allTmon[i][j])
#        meanTmon[i].append(newmean)
    
    meanTmon.append( meanT[i])
#    else: #for files with all timesteps written out
#        meanTmon.append( np.asarray( [monmean(meanT[i][j,:]) for j in range(len(meanT[i]))] ) )


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
#%% 

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

#%% plot last timestep of all zonal gradients in all increasing co2 experiments
# as function of merid grad at last timestep
import statsmodels.api as sm
from scipy import stats

marks=['o','*','^']
cols = ['tab:blue','tab:orange','tab:red']


    
for i in range(216):
    x=[]
    y=[]
    for exp in range(len(allT)):
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
    

#%% meridional vs zonal T gradient ###################################
######################################################################
######################################################################
# lines colored by meanT at time
# feedback params of three boxes given as color of scatters at end of line

from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import matplotlib.colors as colors

exp=0 #co2 amount

#for runs with seasonal variation
#zonalgrad = [ rmean( allTmon[exp][i][0] - allTmon[exp][i][1],13) for i in range(len(allTmon[exp])) ]
#meridgrad = [ rmean( (allTmon[exp][i][0] + allTmon[exp][i][1])/2 - allTmon[exp][i][2], 13) for i in range(len(allTmon[exp])) ]
#for runs without seasonal variation

zonalgrad =  [ allT[i][0] - allT[i][1] for i in range(len(allT)) ]
meridgrad = [ (allT[i][0] + allT[i][1])/2 - (allT[i][2]+allT[i][3])/2 for i in range(len(allT)) ]
#if len(zonalgrad) > 216: #if dealing with data which has random cmip fb at beginning 19
#    zonalgrad = zonalgrad[19:]
#    meridgrad = meridgrad[19:]

fig = plt.figure(figsize=(7,7))
axs = fig.add_subplot()

#normalize to the min and max of all data, continuous colorbar
#norm = plt.Normalize(np.min(meanTmon), np.max(meanTmon))

#discrete colorbar for line color
cmap = plt.cm.get_cmap('gray')
def truncate_colormap(cmap, minval=0.4, maxval=0.9, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = truncate_colormap(cmap)
#norm = BoundaryNorm(np.logspace(
#        np.log10( 1.0), np.log10(10),num=15), #np.max(meanTmon[exp])
#            15)
#norm = plt.Normalize(np.min(meanTmon), 6)
norm = plt.Normalize(0,1800)

#discrete colorbar for feedbacks
#cmapfb = plt.cm.get_cmap('coolwarm', 10)
#normfb = BoundaryNorm(np.linspace(np.min(fb)-0.1,0,10), 10)
cmapfb = plt.cm.get_cmap('coolwarm', 5)
normfb = BoundaryNorm(np.arange(-2.5,2.6,1), 5)
# if using the cmip-ranged sensitivity, these will give different colorbars for each fb
# b1fb = np.linspace(min(r1) - buffer - 0.05, max(r1) + buffer + 
#                    np.ptp(np.linspace(min(r1) - buffer, max(r1) + buffer, 4))/4, 5)
# b2fb = np.linspace(min(r2) - buffer - 0.05, max(r2) + buffer + 
#                    np.ptp(np.linspace(min(r2) - buffer, max(r2) + buffer, 4))/4, 5)
# b3fb = np.linspace(min(r3) - buffer - 0.05, max(r3) + buffer + 
#                    np.ptp(np.linspace(min(r3) - buffer, max(r3) + buffer, 4))/4, 5)
# b4fb = np.linspace(min(r4) - buffer - 0.05, max(r4) + buffer + 
#                    np.ptp(np.linspace(min(r4) - buffer, max(r4) + buffer, 4))/4, 5)

# normfb1 = BoundaryNorm( b1fb, 4)
# normfb2 = BoundaryNorm( b2fb, 4)
# normfb3 = BoundaryNorm( b3fb, 4)
# normfb4 = BoundaryNorm( b4fb, 4)

#plot * at initial value (same for all)
x0 = np.mean([zonalgrad[i][0] for i in range(len(zonalgrad))]) #allT[exp][i][0][0] - allT[exp][i][1][0]
y0 = np.mean([meridgrad[i][0] for i in range(len(zonalgrad))])#(allT[exp][i][0][0]+allT[exp][i][1][0])/2 - allT[exp][i][2][0]
#plt.scatter(x0, y0, marker='*',s=50,color='yellow',zorder=10)
plt.scatter(0, 0, marker='*',s=80,color='green',zorder=10)

zonalgrad -= x0
meridgrad -= y0
for i in range(len(zonalgrad)):
    
    
    #this is a method for coloring the line by dividing it into
    #segments, each of which is colored individually
    #by providing a universal colorbar scale from the "norm" above,
    #all lines will use the same colorbar
    points = np.array([ zonalgrad[i], meridgrad[i] ]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm,alpha=0.8)
    # Set the values used for colormapping
#    lc.set_array(meanTmon[exp][i][1:])
    lc.set_array(np.arange(1800))
    lc.set_linewidth(1)
    line = axs.add_collection(lc)
#    axs.plot( zonalgrad[i], meridgrad[i], color = 'grey', alpha=0.8,
#             path_effects=[pe.Stroke(linewidth=2, foreground='lightsteelblue'), pe.Normal()])
    
    #plot markers indicating the feedback param in each box
    # many reach equil and bunch up all points, while some don't, need way to consistently assign scatters
    spacing=0.02
    x1 = zonalgrad[i][-1]
    x2i = np.argmin( np.abs(zonalgrad[i][:] - (x1 + spacing) ) )
    x2 = zonalgrad[i][x2i]
    x3i = np.argmin( np.abs(zonalgrad[i][:] - (x2 + spacing) ) )
    x3 = zonalgrad[i][x3i]
    x4i = np.argmin( np.abs(zonalgrad[i][:] - (x3 + spacing) ) )
    x4 = zonalgrad[i][x4i]
    y1 = meridgrad[i][-1]
    y2 = meridgrad[i][x2i]
    y3 = meridgrad[i][x3i]
    y4 = meridgrad[i][x4i]
    plt.scatter( x1, y1, marker="s", c=fb[i][0], 
                cmap=cmapfb, norm=normfb, zorder=10, alpha=0.8 )
    plt.scatter( x2, y2, marker="^", c=fb[i][1], 
                cmap=cmapfb, norm=normfb, zorder=10, alpha=0.8 )
    plt.scatter( x3, y3, marker="o", c=fb[i][2], 
                cmap=cmapfb, norm=normfb, zorder=10, alpha=0.8 )
    plt.scatter( x4, y4, marker=">", c=fb[i][3], 
                cmap=cmapfb, norm=normfb, zorder=10, alpha=0.8 )
    
    #plot markers at time intervals
#    times = np.asarray([1,5,10,20,50,100])*12
#    for t in times:
#        plt.scatter( zonalgrad[i][t], meridgrad[i][t], marker=f"${int(t/12)}$",
#                    color='white', s=90)

axs.set_xlim((-2,2))  
axs.set_ylim((-1.,1.))
#fig.colorbar(line, ax=axs,label='Δ Mean T [K]') 
cb = plt.colorbar(ticks=np.arange(-4,2)) 
cb.set_label(label='Regional λ',fontsize=16)
axs.set_xlabel('Δ Zonal gradient [K]',fontsize=16)
axs.set_ylabel('Δ Meridional gradient [K]',fontsize=16)
axs.set_facecolor('whitesmoke')
axs.tick_params(axis='both', which='major', labelsize=14)
axs.tick_params(axis='both', which='minor', labelsize=13)


# draw lines through initial point to divide into quadrants
plt.hlines(0, xmin = -2, xmax= 2, linestyle = '--', color = 'black', alpha=0.7)
plt.vlines(0, ymin = -2, ymax= 2, linestyle = '--', color = 'black', alpha=0.7)

#plt.hlines(y0, xmin = np.min(zonalgrad), xmax= np.max(zonalgrad), linestyle = '--', color = 'black', alpha=0.7)
#plt.vlines(x0, ymin = np.min(meridgrad), ymax= np.max(meridgrad), linestyle = '--', color = 'black', alpha=0.7)


legend_elements = [Line2D([0], [0], marker='s', lw=0, label='Box 1 (West)',
                          markerfacecolor='black', markersize=10),
                   Line2D([0], [0], marker='^', lw=0, label='Box 2 (East)',
                          markerfacecolor='black', markersize=10),
                   Line2D([0], [0], marker='o', lw=0, label='Box 3 (North)',
                          markerfacecolor='black', markersize=10),
                   Line2D([0], [0], marker='>', lw=0, label='Box 4 (South)',
                          markerfacecolor='black', markersize=10)
                   ]
axs.legend(handles=legend_elements, loc='upper right')
#%% investigate fb's of merid strengthening

#print fb of merid grad strengthening
for i in range(len(fb)):
    if meridgrad[i][-1] > meridgrad[i][0]:
#        print( (fb[i][0]+fb[i][1])/2 - (fb[i][2]+fb[i][3])/2 )
        if ( (fb[i][0]+fb[i][1])/2 - (fb[i][2]+fb[i][3])/2 ) < 2:
            
            print(fb[i], (fb[i][0]+fb[i][1])/2 - (fb[i][2]+fb[i][3])/2)

# trop - merid fb
        # in cmip
#0.5366672639637105
#0.7687641804775038
#0.5803922580236145
#0.6170139372152703
#0.7493279967974629
#0.8623636039335876
#0.8619371608400147
#0.9172782021399819
#0.9201802263954566
#0.9639996765078983
#0.8976343638584476
#0.5860014891984611
#0.9322600812559071
#1.0839136467892854
#1.0972673167149085
#1.0846680455555426
#1.0398422252643271
#0.6631008162106455
#0.6297312500945581
#0.5828314665194774
        

#fig.savefig('gradsandtemp.png')
#plt.close(fig)
#%% 
# COLORED GRIDS OF RELATIVE FB PARAM STRENGTHS ZONAL/MERID
# Notes about results of this plot:
# quad 1: more pos box 2 warms box 2, weakens zonal, while more neg box 3 cools box 3, strengthens merid
# quad 2: more pos box 1 warms box 1, strengthen zonal, while more neg box 3 cools box 3, strengthens merid
# quad 3: more neg box 1 cools box 1, weakens zonal, while more pos box 3 warms box 3, weakens merid
# quad 4: more neg box 2 cools box 2, strengthen zonal, more pos box 1 warms box 1, also strengthen zonal

exp=0

#for runs with seasonal variation
#zonalgrad = [ rmean( allTmon[exp][i][0] - allTmon[exp][i][1],13) for i in range(len(allTmon[exp])) ]
#meridgrad = [ rmean( (allTmon[exp][i][0] + allTmon[exp][i][1])/2 - allTmon[exp][i][2], 13) for i in range(len(allTmon[exp])) ]
#for runs without easonal variation
zonalgrad =  [ allT[i][0] - allT[i][1] for i in range(len(allT)) ]
meridgrad = [ (allT[i][0] + allT[i][1])/2 - (allT[i][2]+allT[i][3])/2 for i in range(len(allT)) ]
    
#for seasonal means
#x0 = np.mean([zonalgrad[i][6] for i in range(216)]) #allT[exp][i][0][0] - allT[exp][i][1][0]
#y0 = np.mean([meridgrad[i][6] for i in range(216)])#(allT[exp][0][0][0]+allT[exp][0][1][0])/2 - allT[exp][0][2][0]

#function to check if val1 is within pct% of val2
def pctof(val1,val2,pct=10):
    val1 = abs(val1)
    val2 = abs(val2)
    thresh = val2*pct/100
    up = val2+thresh
    down = val2-thresh
    
    if (val1<=up) & (val1>=down):
        return True
    else:
        return False

#quad 1: x<x0, y>y0. quad 2: x>x0, y>y0. quad 3: x<x0, y<y0. quad 4: x>x0, y<y0
titles=['Zonal weaken & Merid strengthen','Zonal strengthen & Merid strengthen',
        'Zonal weaken & Merid weaken','Zonal strengthen & Merid weaken']
quads = [ [], [], [], [] ]
for i in range(len(fb)):
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
    fb4quad = [ quads[i][j][3] for j in range(len(quads[i]))]
    
    #create stats about relative param strengths, 9 possibilities
    arr = np.zeros((3,3))
#    zonalstr=0 ; zonaleq=0 ; zonalweak=0  
#    meridstr = 0 ; merideq=0 ; meridweak = 0
    for j in range(len(fb1quad)):
        extropfb = np.average([fb3quad[j],fb4quad[j]], weights = [M3,M4])
        tropfb = np.average([fb1quad[j],fb2quad[j]], weights = [M1,M2])
        if (fb1quad[j] > fb2quad[j]): # if box 1 param > box 2 param, col 1
            if pctof( extropfb , tropfb ): #extrop~trop row 2
                arr[1,0] +=1
            elif extropfb < tropfb: #extrop<trop row 1
                arr[2,0] +=1
                if i==2:
                    print(fb1quad[j],fb2quad[j],fb3quad[j],fb4quad[j])
                    print(extropfb - tropfb)
            else: #row 3
                arr[0,0] +=1
        elif fb1quad[j] == fb2quad[j]: #col 2
            if pctof( extropfb , tropfb ): #extrop~trop row 2
                arr[1,1] +=1
            elif extropfb < tropfb: #extrop<trop
                arr[2,1] +=1
            else:
                arr[0,1] +=1
        elif (fb1quad[j] < fb2quad[j]): # if box 3 param > avg of box1&2 col 3
            if pctof( extropfb ,tropfb ): #extrop~trop row 2
                arr[1,2] +=1
            elif extropfb < tropfb: #extrop<trop
                arr[2,2] +=1
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
    ax[i].set_xticklabels(['West>East','West~East','West<East'])
    ax[i].set_yticks(np.arange(0, 3, 1))
    ax[i].set_yticklabels(['ExTrop>Trop','ExTrop~Trop','ExTrop<Trop'])
    
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
    ax[i].set_title(f"{titles[i]} n={len(quads[i])}", weight='bold')
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

colors = ['tab:blue','tab:purple','tab:orange','tab:red']
sens = []
co2 = [8]

#create dataframe with feedbacks, clim sens, and net feedback (estimated from regression)
df = pd.DataFrame({'Box Feedbacks':fb, 'Sensitivity':np.zeros(len(fb)), 
                   'Net Feedback':np.zeros(len(fb)), 'Max Zonal Gradient':np.zeros(len(fb)),
                   'Time of Max Z Grad':np.zeros(len(fb)), 'Init Z Gradient':np.zeros(len(fb)),
                   'Final Z Gradient':np.zeros(len(fb)), 'Time of Equal Z Grad':np.zeros(len(fb)), 
                   'Min Meridional Gradient':np.zeros(len(fb)),
                   'Time of Max M Grad':np.zeros(len(fb)), 'Init M Gradient':np.zeros(len(fb)),
                   'Final M Gradient':np.zeros(len(fb)), 'Time of Equal M Grad':np.zeros(len(fb))})

for c in range(len(co2)):
    sens.append([])
    zonalgrad = [ allTmon[c][i][0] - allTmon[c][i][1] for i in range(len(allTmon[c])) ]
    meridgrad = [ (allTmon[c][i][0]+allTmon[c][i][1])/2 - (allTmon[c][i][2]+allTmon[c][i][3])/2 for i in range(len(allTmon[c])) ]
    for i in range(len(fb)):
        
        T1 = allTmon[c][i][0]
        T2 = allTmon[c][i][1]
        T3 = allTmon[c][i][2]
        T4 = allTmon[c][i][3]
        
        
        toa_b1 = S1*(1-alpha1) - (B*(t01-273.15) + A) + co2[c] + (T1 - t01) * fb[i][0]
        toa_b1 = toa_b1*area1 / (area1+area2+area3+area4)
        toa_b2 = S2*(1-alpha2) - (B*(t02-273.15) + A) + co2[c] + (T2 - t02) * fb[i][1]
        toa_b2 = toa_b2*area2 / (area1+area2+area3+area4)
        toa_b3 = S3*(1-alpha3) - (B*(t03-273.15) + A) + co2[c] + (T3 - t03) * fb[i][2]
        toa_b3 = toa_b3*area3 / (area1+area2+area3+area4)
        toa_b4 = S4*(1-alpha4) - (B*(t04-273.15) + A) + co2[c] + (T4 - t04) * fb[i][3]
        toa_b4 = toa_b4*area4 / (area1+area2+area3+area4)
        
        #linear regression to determine x intercept for each fb (=clim sens)
        # netTOA = slope*meanT + y-int
        # climsens = - y-int/slope
        x0 = sm.add_constant(meanTmon[c][i]) #add column for intercept
        
        lr = sm.OLS(toa_b1+toa_b2+toa_b3+toa_b4,x0,missing='drop').fit()
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
        df['Max Zonal Gradient'][i] = np.max(zonalgrad[i])
        df['Init Z Gradient'][i] = zonalgrad[i][0]
        df['Final Z Gradient'][i] = zonalgrad[i][-1]
        df['Time of Max Z Grad'][i] = np.argmax(zonalgrad[i])
        #zonal thermosat duration 
        thermdur = np.argmin(np.abs(zonalgrad[i][0] - zonalgrad[i][5:])) + 5
        if thermdur == 5: #if gradient only increases in strength
            df['Time of Equal Z Grad'][i] =  np.nan
        else:
            df['Time of Equal Z Grad'][i] = thermdur
        
        df['Min Meridional Gradient'][i] = np.min(meridgrad[i])
        df['Init M Gradient'][i] = meridgrad[i][0]
        df['Final M Gradient'][i] = meridgrad[i][-1]
        df['Time of Max M Grad'][i] = np.argmax(meridgrad[i])
        df['Time of Equal M Grad'][i] = np.argmin(np.abs(meridgrad[i][0] - meridgrad[i][5:])) + 5 
        
        
        #plt.scatter( meanTmon[c][i], toa_b1+toa_b2+toa_b3 ,color = colors[c], s=5)
        #plt.scatter( sens[c][i],0)
        #add regression line to visualize x intercept
        #plt.plot( np.arange(0,30), slope*np.arange(30)+yint, color='grey',linewidth=0.5,alpha=0.6)

#set rows to nan if
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


var = 'Min Meridional Gradient'
#filter out nans and extreme values
isdata = np.where( (df[var].notnull()) & (np.abs(df['Min Meridional Gradient'])<10) )[0]
marks=['o','x','^','*']
offset=[0,0,0,0] #[-0.2,-0.1,0.1,0.2]



for f in range(3): #for each fb
    
    #individual regional fb params
    x = np.asarray( [ i[f] for i in df['Box Feedbacks'].iloc[isdata][:] ] )
    labels=['fb1','fb2','fb3','fb4']
    
    y = df[var].iloc[isdata]
    
    #regression to determine linear fit
    x0 = sm.add_constant(x) #add column for intercept
            
    lr = sm.OLS(y,x0,missing='drop').fit()
    
    plt.scatter( x+offset[f], y , marker=marks[f], label = f"{labels[f]} {lr.rsquared.round(decimals=3)}")
    
    #differences of regional fb params
    x = np.asarray( [ i[f]-i[f-1] for i in df['Box Feedbacks'].iloc[isdata][:] ] )
    labels=['fb1-fb3','fb2-fb1','fb3-fb2']
    
    x0 = sm.add_constant(x) #add column for intercept
            
    lr = sm.OLS(y,x0,missing='drop').fit()
    
    plt.scatter( x, y , label = f"{labels[f]} {lr.rsquared.round(decimals=3)}")
    
#meridional gradient as predictor
x = np.asarray( [ (i[0]+i[1])/2 - i[2] for i in df['Box Feedbacks'].iloc[isdata][:] ] )
y = df[var].iloc[isdata]

#regression to determine linear fit
x0 = sm.add_constant(x) #add column for intercept
        
lr = sm.OLS(y,x0,missing='drop').fit()

plt.scatter( x, y , label = f"(fb1+fb2)/2 - fb3 {lr.rsquared.round(decimals=3)}")
    
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

#%% read in cmip6-fb runs
meanT = np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/cmip6.kernelfb.8wm2.meanT.150.npy')
allT = np.load('C:/Users/Scott/Documents/Python Scripts/boxmodel/cmip6.kernelfb.8wm2.allT.150.npy')

#%% PLOTS FOR CMIP MODELS
###################################################################################
#mean T
for i in range(len(meanT)):
    plt.plot( meanT[i] , color=mcolors[i])

plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Years') 
plt.ylabel('mean T')
    
#%% zonal
plt.figure()
zonalgrad = [ allT[i][0] - allT[i][1] for i in range(len(allT)) ]

cmap = plt.get_cmap('bwr')
wcolors=cmap(np.linspace(0,1,len(changed)))

for i in range(len(allT)):
    plt.plot( zonalgrad[i], color=wcolors[i] )

#plt.ylim([0,5])
plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Years') 
plt.title('West-East Pac',fontsize=15)

#%% meridional
plt.figure()
extrop = [ (allT[i][2]*M3 + allT[i][3]*M4)/(M3+M4) for i in range(len(allT)) ]
meridgrad = [ (allT[i][0] + allT[i][1])/2 - extrop[i] for i in range(len(allT)) ]

for i in range(len(allT)):
    plt.plot( meridgrad[i], color=wcolors[i] )

plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Years') 
plt.title('Trop-ExTrop Pac',fontsize=15)

#%% zonal vs meridional
zonalgrad = [ allT[i][0] - allT[i][1] for i in range(len(allT)) ]
extrop = [ (allT[i][2]*M3 + allT[i][3]*M4)/(M3+M4) for i in range(len(allT)) ]
meridgrad = [ (allT[i][0] + allT[i][1])/2 - extrop[i] for i in range(len(allT)) ]

for i in range(len(allT)):
    plt.plot( zonalgrad[i], meridgrad[i], color=wcolors[i] )
    
plt.xlabel('Zonal gradient [K]', fontsize=14)
plt.ylabel('Meridional gradient [K]', fontsize=14)
#%% all boxes
    
plt.title("Temperatures")
expnum=len(allT)
for i in range(expnum):
    plt.plot(allT[i][0],color=mcolors[i])
    plt.plot(allT[i][1],color=mcolors[i])
    plt.plot(allT[i][2],color=mcolors[i])
    plt.plot(allT[i][3],color=mcolors[i])
    plt.plot(allT[i][4],color=mcolors[i])

plt.annotate("Trop W",xy=(0,allT[0][0][0]-0.3))
plt.annotate("Trop E",xy=(0,allT[0][1][0]-0.3))
pltlen=int(len(allT[0][0])-0.2*len(allT[0][0]))
plt.annotate("Ex Trop N",xy=(pltlen,allT[0][2][pltlen]))
plt.annotate("Ex Trop S",xy=(pltlen,allT[0][3][pltlen]))
plt.annotate("UnderCurrent",xy=(0,allT[0][4][100]))

plt.xticks(ticks=np.linspace(0,len(t1),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Years') 
plt.legend()

#%% compare final temps between box model and cmip

cmipwest = [
5.317439296937615 ,
4.23576547479832 ,
2.6471753811465573 ,
3.849355067285773 ,
3.425718791598172 ,
8.295928535409585 ,
5.0445861573644155 ,
5.037036953987397 ,
4.451714457310835 ,
4.516521409353405 ,
3.1773397714254723 ,
5.688561393368199 ,
2.7870969524269946 ,
1.540584811941265 ,
4.014314931236993 ,
3.767700554828897 ,
2.5873316022686725 ,
4.975759079869772 ,
5.023702618897059 ,
6.031064870762728
]
cmipeast = [
5.722134821570752 ,
4.659441132950339 ,
2.5648264217107246 ,
4.492496669348431 ,
4.005267701904192 ,
11.319327213000319 ,
6.743521430444507 ,
6.692437543073716 ,
6.494159812481334 ,
4.897337321271708 ,
3.4872853879767134 ,
7.139615912856968 ,
4.504067971457132 ,
2.982379332241905 ,
5.474508120504309 ,
3.974414266607462 ,
2.9107233888105157 ,
5.1252260572763815 ,
7.457046282078699 ,
6.963737576202504
        ]
cmipnorth = [
        5.331836184817201 ,
4.468546284512108 ,
3.8911974038622956 ,
4.525793472139346 ,
3.432926836302463 ,
7.959147181031481 ,
4.488680361414968 ,
4.610141716960852 ,
3.8839003726047125 ,
4.027295917684933 ,
3.2573681544033217 ,
6.078470813270363 ,
3.735427905284741 ,
3.347298561904065 ,
3.920054025990807 ,
4.574304547636112 ,
3.664950248405717 ,
4.449354036387424 ,
4.554032053662816 ,
5.952702177875024
        ]

cmipsouth = [
        4.043170495033338 ,
3.291436718370807 ,
3.117575628965767 ,
3.328508418188466 ,
2.978641776204887 ,
9.73927930331121 ,
4.7557999620716656 ,
5.395266573407693 ,
4.571963991543822 ,
3.9129403982319584 ,
2.457714849804559 ,
5.405994124907365 ,
2.6220800800182196 ,
2.480441594036926 ,
3.484375286190383 ,
3.253633833762891 ,
2.074874054742093 ,
4.604882457453889 ,
4.5627095957296655 ,
5.161978506985018
        ]
wdiff=[]
ediff=[]
ndiff=[]
sdiff=[]

for i in range(len(fb)):
    wdiff.append( allTmon[0][i][0][-1] - t01 - cmipwest[i])
    ediff.append( allTmon[0][i][1][-1] - t02 - cmipeast[i])
    ndiff.append( allTmon[0][i][2][-1] - t03 - cmipnorth[i])
    sdiff.append( allTmon[0][i][3][-1] - t04 - cmipsouth[i])

cmap = plt.get_cmap('jet')
colors=cmap(np.linspace(0,1,len(fb)))    
for i in range(len(fb)):
    plt.scatter( 1,wdiff[i], color=colors[i])
    plt.scatter( 2, ediff[i], color=colors[i])
    plt.scatter( 3, ndiff[i], color=colors[i])
    plt.scatter( 4, sdiff[i], color=colors[i])
    plt.plot( [1,2,3,4], [wdiff[i],ediff[i],ndiff[i],sdiff[i]], color=colors[i])
plt.grid()
plt.title('Box model - CMIP6 regional temp difference')

#%% box model - cmip6 mean warming in each region
wmean = np.mean(cmipwest)
emean = np.mean(cmipeast)
nmean = np.mean(cmipnorth)
smean = np.mean(cmipsouth)

wdiff=[]
ediff=[]
ndiff=[]
sdiff=[]

for i in range(len(fb)):
    wdiff.append( allT[i][0][-1] - t01 - wmean)
    ediff.append( allT[i][1][-1] - t02 - emean)
    ndiff.append( allT[i][2][-1] - t03 - nmean)
    sdiff.append( allT[i][3][-1] - t04 - smean)

for i in range(len(fb)):
    plt.scatter( 1,wdiff[i], color=colors[i])
    plt.scatter( 2, ediff[i], color=colors[i])
    plt.scatter( 3, ndiff[i], color=colors[i])
    plt.scatter( 4, sdiff[i], color=colors[i])
    plt.plot( [1,2,3,4], [wdiff[i],ediff[i],ndiff[i],sdiff[i]], color=colors[i])
plt.grid()
plt.title('Box model - CMIP6 mean regional temp difference')

#%%  Relative warming in each box by certain year

plt.figure(1)
plt.title("ΔTemperatures")
cmap = plt.get_cmap('Blues')
#wcolors=cmap(np.linspace(0,1,len(changed)))
#cmap = plt.get_cmap('Reds')
#ecolors=cmap(np.linspace(0,1,len(changed)))
#cmap = plt.get_cmap('Greens') 
#scolors=cmap(np.linspace(0,1,len(changed))) 
#cmap = plt.get_cmap('Purples')
#ncolors=cmap(np.linspace(0,1,len(changed)))
#cmap = plt.get_cmap('Oranges') 
#ucolors=cmap(np.linspace(0,1,len(changed)))  
for i in range(len(fb)):
#    plt.plot(allT[i][0] - t01,color=wcolors[i], alpha= 0.8)
#    plt.plot(allT[i][1] - t02,color=ecolors[i], alpha= 0.8)
#    plt.plot(allT[i][2] - t03,color=ncolors[i], alpha= 0.8)
#    plt.plot(allT[i][3] - t04,color=scolors[i], alpha= 0.8)
#    plt.plot(allT[i][4] - t05,color=ucolors[i], alpha= 0.8)
    plt.plot(allT[i][0] - t01,color='tab:blue', alpha= 0.6)
    plt.plot(allT[i][1] - t02,color='tab:purple', alpha= 0.6)
    plt.plot(allT[i][2] - t03,color='tab:red', alpha= 0.6)
    plt.plot(allT[i][3] - t04,color='tab:orange', alpha= 0.6)
    plt.plot(allT[i][4] - t05,color='tab:green', alpha= 0.6)
    plt.plot(allT[i][5] - t06,color='tab:pink', alpha=0.6)
    plt.plot(allT[i][6] - t07,color='tab:cyan', alpha=0.6)


plt.xticks(ticks=np.linspace(0,len(allT[0][0][:]),6),labels=np.linspace(0,int(years),6).round())
plt.xlabel('Years',fontsize=14) 
plt.ylabel('ΔT',fontsize=14)
plt.legend(labels=['west','east','north','south','under','high north','high south'],fontsize=14)
# plt.xlim([0,24])
# plt.ylim([0,5])

#%% sensitivity of Ah or Aw
cmap = plt.get_cmap('coolwarm')
wcolors=cmap(np.linspace(0,1,len(changed)))

zonalgrad = [ allT[i][0] - allT[i][1] for i in range(len(allT)) ]
extrop = [ (allT[i][2]*M3 + allT[i][3]*M4)/(M3+M4) for i in range(len(allT)) ]
meridgrad = [ (allT[i][0] + allT[i][1])/2 - extrop[i] for i in range(len(allT)) ]


awvalues = [      0.,  200000.,  400000.,  600000.,  800000., 1000000.,
       1200000., 1400000., 1600000., 1800000., 2000000.]

for i in range(len(allT)):
    plt.scatter( awvalues[i], zonalgrad[i][-1] , color = wcolors[i] )
    
plt.xlabel('Aw', fontsize=14)
plt.ylabel('150yr Zonal Gradient [K]', fontsize=14)
plt.title('Zonal gradient as function of Aw value', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

#%% sensitivity to under volume

zonalgrad = [ allT[i][0] - allT[i][1] for i in range(len(allT)) ]
extrop = [ (allT[i][2]*M3 + allT[i][3]*M4)/(M3+M4) for i in range(len(allT)) ]
meridgrad = [ (allT[i][0] + allT[i][1])/2 - extrop[i] for i in range(len(allT)) ]

labels = ["1/8", "1", "8"]
plt.scatter(zonalgrad[0][0], meridgrad[0][0], marker='*', color='red', zorder=10, s=130)
for i in range(len(labels)):
    plt.plot( zonalgrad[i], meridgrad[i], label = labels[i], linewidth=2)
    
plt.xlabel('Zonal gradient [K]', fontsize=19)
plt.ylabel('Meridional gradient [K]', fontsize=19)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.grid()
plt.legend(fontsize=20)
    