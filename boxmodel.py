# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:11:59 2022

@author: Scott
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import multiprocessing


##########################################################
################### PARAMETERS ###########################
##########################################################

timesteps=6000*150 #second number is total years
dt=5256 #seconds, 1/6000 of year
# print(f"timestep dt ={dt/60:.2f} minutes")
totaltime=timesteps*dt
years=totaltime/3600/24/365
print(f"total time = {years:.2f} years")
Cp = 4.186e3 # J/(kg K) specific heat of ocean water
rho = 1027 # kg/m3 density of sea water at surface
rhoair = 1.225 # kg/m3 dens of air
p_air=100000 # Pa air pressure

epsilon = 0.48 #0.48 # ratio of flow out
warea = 15774945387831 #surface area of box 1, west
earea = 15774945387831
narea = 33306750009756
sarea = 28541763159964
hnarea = 12430460163633
hsarea = 16826744191172
h1=50 #60 #m, depth of box 1 (west)
h2=50 #20 #50 #m, depth of box 2 (east)
h3=200 #m, depth of box 3 (north)
h4=200 #m, depth of box 5 (south)
h6=3000 # m, depth of box 6 (high north)
h7=3000 # m, depth of box 7 (high south)
M1= warea * h1  #m3, vol of box 1 (west)
M2= earea * h2  #m3, vol of box 2 (east)
M3= narea * h3  #m3, vol of box 3  (north) 
M4= sarea * h4  #m3, vol of box 4 (south)
M6 = hnarea * h6 # vol of box 6 (high north)
M7 = hsarea * h7 # vol of box 7 (high south)
#for 20-50N 160-240E =24031081172332.445*200=4806216234466489.0 ~ 4.8e15m3
#was using 4e15
underarea = warea + earea + narea + sarea
underdepth = 150 #150 for cmip-like...?
M5=(underarea * underdepth) #9.5e15 #4e15 #9.6e15 #8e15 #m3, vol of box 5 (undercurrent)

Aw=1e6 # m3/(s K) walker coupling param
Ah=4e6 # m3/(s K) hadley coupling param


#constants for MSE div
cp_air=1005; #J/kg/K
L=2250000; #J/kg or 2260000
ps=100000; #Pa
RH=0.8; # relative humidity in tropics
epsil=0.622; # ration of mass or vapour to dry air
mse_gamma = 0.001581 # W/m2 / J/kg
mse_gamma_w = -0.002 # era5 sst: -0.007  CMIP6: -0.002
mse_gamma_e = -0.002 # era5 sst: -0.002 CMIP6: -0.002
mse_gamma_n = -0.005 # era5 sst: -0.002  CMIP6: -0.005
# mse_gamma_n_trop = -0.001
mse_gamma_s = -0.008 # era5 sst: -0.005 CMIP6: -0.008
# mse_gamma_s_trop = -0.001
mse_gamma_hn = -0.0002 # era5 sst: -0.0002 CMIP6: -0.0002
mse_gamma_hs = -0.0002 # era5 sst: -0.0002  CMIP6: -0.0002
# mse_int = -9.56 #-9.56 #intercept for mse regression
#regression done on Cheyenne notebook mse_aht_regression.ipynb
mse_int_w= 0#166.087662215735
mse_int_e= 0#126.31678504385408
mse_int_n= 0#-65.70420947798185
mse_int_s= 0#-59.4926645914937
mse_int_hs= 0#42.60663756316386
mse_int_hn= 0#75.26594248197749

#the following based on CERES data
#boxes used for ceres radiation data: 
# 1: 8-8N/S 120-200E  2: 8-8N/S 200-280E   3: 8-50N 165-230E 4: 8-40S 200-280E

# ERA5 SW+LW
swlw1 = 73 # W/m2
swlw2 = 55
swlw3 = 18
swlw4 = 26
swlw6 = -81
swlw7 = -49

areas=[warea,earea,narea,sarea,hnarea,hsarea]


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

##############################################################
#####################SETUP################################
##############################################################
  

# fb from lambda = ( F - dTOA ) / dT where F is from Huang 2016     
fb = [
[-0.434172015740533 ,0.12190893473122326 ,-0.2096268338325976 ,-0.4442660878000541 ,0.5329596952913375 ,-0.43393210970991286],
[-1.246172338268635 ,-0.4342935976281434 ,-0.7456314588325751 ,-1.219241804932912 ,0.11797175974664524 ,-0.3990850293325592],
[1.4275150427197016 ,1.9022093554344097 ,-0.1154916149368109 ,-0.38799025472476123 ,0.27417691080271706 ,-0.6622826654480054],
[-0.882943442382047 ,1.5147274340123165 ,-0.6808286045289191 ,0.6922100794478377 ,0.1867248113547375 ,-1.1705502417876827],
[-0.986915321482554 ,1.8232078257976134 ,-0.696991688961643 ,0.9500393927476528 ,0.46937273245113065 ,-0.9281526423771276],
[-0.9726159696877374 ,0.9032158539779798 ,-1.0144133550867431 ,0.05897273786872013 ,0.26914855489845624 ,-0.8733179901242596],
[0.022387824617162003 ,-0.07212767511357789 ,-0.7420183136623971 ,-0.7966121170293152 ,-0.22018568443891207 ,-0.6504293075884104],
[-0.01595345900537056 ,-0.17170958341438 ,-0.8113521140646989 ,-0.9010505531989306 ,-0.1948693909333026 ,-0.868595165041488],
[-0.8284633614500002 ,-0.6771868168909833 ,-1.3180226808069844 ,-1.3431552761203238 ,0.7484920908145248 ,-0.6846304942371632],
[-1.201779377405116 ,-0.3439305613902023 ,-1.2515033041902963 ,-1.2698058951329256 ,0.0853168942782505 ,-0.021732721992505335],
[-1.5734230407917995 ,-0.7318177836020384 ,-1.1723013338970052 ,-1.3951548389781836 ,0.5426253610186503 ,-1.156921467842894],
[-1.9653230134044881 ,-0.7642116553111263 ,-1.227870375860378 ,-1.8887760775287645 ,1.2406590846756993 ,-1.601246361723613],
[-0.5979475851549619 ,0.22245836033670782 ,-1.316057132670607 ,-1.2460213671315759 ,-0.1037358815190191 ,-0.7556714069526324],
[-0.6887184447604296 ,-0.15183811796815547 ,-1.0766306400865238 ,-1.5272305859121746 ,0.24922692561229454 ,-1.0707269722503616],
[0.24321991683545693 ,1.0390317539778455 ,-0.8872904812266061 ,-0.36203896488633447 ,0.5492790580335682 ,-0.662313513107862],
[-0.30496733591184516 ,0.17934971357125215 ,-0.19769181165045974 ,-0.3935281172791597 ,0.558654499106031 ,-0.4047196515543394]
]


###########################################################
#### generate sensitivity feedback parameters #############
###########################################################

fb = []

step = 1
fb_low = -3.0
fb_high = 2.0

for i in np.arange(fb_low,fb_high+step,step):
    for j in np.arange(fb_low,fb_high+step,step):
        for k in np.arange(fb_low,fb_high+step,step):
            # fb.append([i,j,k,k]) #using same param for north/south
            for s in np.arange(fb_low,fb_high+step,step):
                fb.append([i,j,k,s,-1.2,-1.2])

# remove net positive fb combinations to reduce sim number
# c=0
# remlist=[]
# for i,val in enumerate(fb):
#     weightfb = np.average( val, weights=[warea,earea,narea,sarea,hnarea,hsarea])
#     if weightfb>0:
#         c+=1
#         remlist.append( val )

# for val in remlist:
#     fb.remove(val)
    


##########################################################################
####### Use full time series of CMIP feedbacks from select models ########
##########################################################################
usefullfb = False
if usefullfb:
    print("USING TIME SERIES FEEDBACKS FROM CMIP")
    # fb = np.load( "/home/sknapp4/boxmodel/cmipfullfeedbacks.npy" )
    fb = np.load( '/home/sknapp4/jup/inferredlambdas_huangf.npy' )
    # fb = np.load( '/home/sknapp4/jup/inferredlambdas_globalf.npy' )

###########################
#CO2 FORCING AS A CONSTANT EVERYWHERE
co2 = [8] #W/m2 4~2xCO2, 8~4xCO2, 12~8xCO2


#########################################
#### Use TOA anoms from CMIP6 instead of forcing & feedbacks ####
#########################################
usetoaanoms = False
if usetoaanoms:
    print('USING CMIP6 TOA ANOMS, FORCING SET TO 0 AND FEEDBACKS OFF')
    fb = np.load( "/home/sknapp4/boxmodel/cmipfulltoaanom.npy" )
    co2=[0]

#######################################    
### Use dOHT from CMIP6 #############
#######################################

highlatcmipoht = False
lowlatcmipoht = False # all 4 low lat boxes
if highlatcmipoht:
    print('USING CMIP6 dOHT for High Lats')
    doht = np.load( "/home/sknapp4/boxmodel/cmipfulldoht.npy" )
    oht0 = np.load("/home/sknapp4/boxmodel/cmipoht0.npy")

#######################################    
### Use AHT from CMIP6 #############
#######################################

highlatcmipaht = False
lowlatcmipaht = False # all 4 low lat boxes
if highlatcmipaht:
    print('USING CMIP6 AHT for High Lats')
    cmipaht = np.load( "/home/sknapp4/boxmodel/cmipfullaht.npy" )
    
    
#######################################################
######## Pattern feedback parameterization ############
#######################################################

fb_adapt_bool=False #make this true to use feedbacks which change based on local and warm pool temp anoms

if fb_adapt_bool:
    print("FB_ADAPT ON")

#######################################################
###### Initialize with CMIP6 average column temps #####
#######################################################

cmip6params = False # 

if not cmip6params:
    print("ORA5 PARAMS")
else:
    print("CMIP6 INDIVIDUAL FBs")
    
#########################################################
####### Use SST estimation for AHT calculations #########
#########################################################
ahtsst = False

if ahtsst:
    print("Using SST estimates for AHT calculations")

#########################################################
########### Subtract mean AHT anom ######################
#########################################################
submeanAHT = False

if submeanAHT:
    print("Subtracting mean AHT anom from each box")
    
#########################################################
########### Regionally Unique Forcing ###################
#########################################################
regionalforcing = False
# if regionalforcing:
#     print( "Regionally unique forcing")
    
#########################################################
######### Prescribed linearly changing FB ###############
#########################################################
uselinearfb = False
if uselinearfb:
    print("East/North/South linear lambdas prescribed")
    linearfb = np.load( "/home/sknapp4/boxmodel/linearfb.npy" )
    
### To fix AHT/OHT at initial values, enable ###
aht_fixed = True # 
if aht_fixed:
    print("AHT is FIXED AT INITIAL VALUE")
oht_fixed = True
if oht_fixed:
    print("OHT is FIXED AT INITIAL VALUE")
#####################################################################################################
#####################################################################################################
#####################################################################################################


# if fb_adapt_bool:
#     if len(fbadapt)!=len(fb):
#         print('FB_ADAPT AND FB NOT SAME LENGTH')


# initial avg col temps from PI CMIP6 members with depths [50,50,200,200,3000,3000] 
t0all = [
    [301.7823964860704 ,298.9543443891737 ,289.6186779120029 ,292.1675000215188 ,274.4614154884855 ,276.3111957458081],
[299.77784067789713 ,298.52013901604545 ,290.34923102550016 ,292.28415532234385 ,274.4497963789339 ,276.5014954283003],
[300.8722472599469 ,297.3982405200878 ,289.4689972680335 ,291.01664256680914 ,274.1640570226855 ,275.454593593134],
[300.4808923509386 ,298.94082043965653 ,290.39547244165504 ,292.3385514231999 ,274.1512664885632 ,275.4458441347829],
[300.4939725240071 ,298.8255966610378 ,290.33149393050525 ,292.33255438278775 ,274.1430366365861 ,275.45969317255845],
[300.30396563212076 ,299.2101713392469 ,290.769508410103 ,292.8838346287984 ,274.2847216131837 ,275.8206348137036],
[300.2974859758506 ,297.27342554304886 ,289.2280148866409 ,290.94668505354434 ,274.13283440549736 ,275.83670261952676],
[300.523366862521 ,297.36158044706866 ,289.45957139300674 ,290.9875890266957 ,274.232593599123 ,276.31556923233956],
[299.9921600515192 ,299.2186976259405 ,289.2268474951023 ,292.12912051968453 ,273.841191003295 ,275.78734205774947],
[301.19351272583003 ,298.68130683898926 ,289.7200891494751 ,292.1321998119354 ,273.82583599338926 ,276.30425701777136],
[301.30081277100936 ,298.6683387217314 ,289.52962771906994 ,292.79178510145704 ,275.02662590817954 ,277.3909260459669],
[301.19324890219644 ,297.7487270811329 ,289.807087615042 ,292.635448028102 ,275.2120520441606 ,276.9368514866852],
[301.7246050895528 ,297.2428657612902 ,289.1105754225221 ,291.5400579348002 ,274.8221165544241 ,276.49713333763594],
[300.8796690433583 ,297.2137925330629 ,288.9550953381682 ,291.4440634427005 ,275.14250150480177 ,276.8254539273849],
[299.48629057142466 ,297.8337741004096 ,289.8837973014541 ,292.595757610983 ,273.9839799502475 ,275.3298463618498],
[301.1018233994925 ,297.2774890053321 ,289.2051988725732 ,291.49548228561605 ,274.2971459967897 ,275.9824627281643]
]    

# ORA5 1979-1999 mean ocean temps of columns given depths [50,50,200,200,3000,3000]
t01 = 301.93304481506345
t02 = 298.8521331787109
t03 = 289.13522090911863
t04 = 292.411266708374
t05 = 290.64704582634755 #weighted mean of t03 and t04
t06 = 274.94597067832945
t07 = 275.6185773849487


swlwall = [
[75.85098964407727 ,62.45114833806736 ,18.09656504842856 ,33.800244007315115 ,-79.59317235609092 ,-50.01488067914038],
[67.70023526680298 ,53.788037087301554 ,15.38379974755538 ,34.02516543356545 ,-94.43642737179204 ,-54.515018042254866],
[76.86686080903941 ,40.2291096793984 ,16.514259342762475 ,27.613672525380263 ,-82.38157428244406 ,-50.54251774737988],
[81.68762126126872 ,53.2401484764757 ,23.41986526403586 ,31.898198821654503 ,-86.12247252078394 ,-54.74145267319065],
[81.95549191051386 ,49.49990907595499 ,23.47560297729298 ,30.67467053712365 ,-87.92313562172292 ,-55.54324964373535],
[77.08828207914516 ,51.5690800428283 ,23.495274225782705 ,36.06842057224215 ,-84.06261147898486 ,-51.709489437818206],
[63.396579964714654 ,62.33290625392814 ,19.846233697448127 ,34.2171631412483 ,-80.87753230229333 ,-49.286038887897135],
[63.0870885128407 ,62.41107400234954 ,19.706823107799142 ,33.81784447024527 ,-80.89698466618124 ,-47.518781345311275],
[74.90213400048441 ,60.62626808299125 ,21.354642315734502 ,36.30974579224132 ,-88.35079926250187 ,-51.25787679798444],
[76.70657017042018 ,58.88323965195152 ,22.275820036683545 ,35.89727817054483 ,-86.6524723282361 ,-52.3554035660751],
[50.93194016639456 ,41.34654478886488 ,12.406740383047614 ,26.75098853477084 ,-73.77726013370979 ,-40.91045997131314],
[57.20883431395939 ,49.29559771852414 ,15.887530550226996 ,26.500004257075584 ,-78.42834808805306 ,-45.87746811459292],
[86.53270239030296 ,49.90396566281195 ,16.762432175952576 ,31.637026645803466 ,-86.64668994344477 ,-57.46821331140644],
[87.69110443966599 ,46.22500292485712 ,15.105296546184409 ,33.145268980882065 ,-89.2044662713957 ,-53.176445309368155],
[81.1200719728292 ,44.83038455710268 ,21.03061227504526 ,36.797072464654214 ,-86.4764832497398 ,-48.58234880412672],
[74.48527618577286 ,58.37590129183735 ,20.421513352507482 ,31.730215838898058 ,-80.20015375610815 ,-50.345142273294044]
]



shfall = [ #pi SHF data from cmip models with ocean data
    [-18.345491893072335 ,-61.77735052521976 ,0.0598807372348222 ,-4.985566484434844 ,6.923401204825725 ,-1.3702312912949675],
[-23.325774438723766 ,-62.43871406234904 ,6.372925821973181 ,-4.626419056834401 ,8.536048751455983 ,5.1354803030857905],
[-28.43636340671695 ,-49.65760674842955 ,2.1479584692855966 ,-4.585593619319396 ,7.605054073704006 ,4.345257053646339],
[-24.692342736255487 ,-61.895519475634565 ,-0.6775272570155317 ,-4.08242992220539 ,6.888114079589415 ,5.278852553770192],
[-26.356040853795406 ,-59.774973670220206 ,-0.9039961977687373 ,-4.092326237387963 ,6.91789708260174 ,5.608981650551682],
[-25.677162527263594 ,-59.9555410392467 ,2.575198976045017 ,-5.572244257650977 ,7.708475722425149 ,1.4577659347054794],
[-21.324496347423775 ,-60.15184708188238 ,3.302077585555034 ,-4.5001176962771785 ,11.411948402949232 ,-0.4893414304619921],
[-21.262305641729004 ,-59.04420018764939 ,2.53016000215633 ,-3.904341771625689 ,12.32317491282049 ,1.049678401360943],
[-27.134055423275157 ,-69.78744431972439 ,7.437643372221328 ,0.4487044895502717 ,2.128677360391254 ,6.304485920511113],
[-18.883693569683565 ,-73.98320159221053 ,2.7283970518892335 ,-5.342711778193299 ,8.954937084353155 ,10.534704627912088],
[-13.814929184398991 ,-45.495070113976645 ,3.1431942390345435 ,-6.337758398640625 ,9.053846342442169 ,8.695967971538035],
[-13.205359073694632 ,-56.514660271933 ,2.550944673558755 ,-5.621365210703401 ,7.177186322031724 ,4.720475326955284],
[-16.823783809324723 ,-69.27321454166434 ,6.353261553046035 ,0.6308506476070568 ,6.896529769845501 ,0.7521175002109983],
[-24.419268120967025 ,-66.13099519483556 ,7.009047710590703 ,-1.9054166400040689 ,6.701831805647726 ,0.5206061922120245],
[-24.92362875618716 ,-64.00734100036549 ,1.7546562596887918 ,-5.153981048544215 ,6.597869308714509 ,5.127420152483011],
[-20.797514858995346 ,-67.34961595025229 ,1.3263435692935046 ,-5.989751591109203 ,6.336432825779622 ,4.652933931000573]
]


################################################################
def atmocntuning( t01,t02,t03,t04,t05,t06,t07, mse_gamma_w,mse_gamma_e,mse_gamma_n,mse_gamma_s,mse_gamma_hn,mse_gamma_hs,
                 mse_int_w,mse_int_e,mse_int_n,mse_int_s,swlw1,swlw2,swlw3,swlw4,swlw6,swlw7,i):
    ##### TUNING TO MATCH ANY SW + LW & INITIAL TEMPS #####
    ## use t0s, TOA, mse int from era5/cmip6 so all atmos is known, get Ah, Aw values
    ## then,TOA and ocean energy terms are known, so can solve for (tune) mse_ints to close budget
    ## this way model starts in perfect balance
    
    # q=Ah*( np.average([t01,t02],weights=[M1,M2]) - np.average([t03,t04],weights=[M3,M4])) + Aw*(t01-t02)
    # q3 = q*M3/(M3+M4)
    # q4 = q-q3
    
    # get SST estimates from ocean temps. from ERA5 notebook on casper
    # sstfromocean1 = t01*0.9641 + 10.88
    # sstfromocean2 = t02*0.8688 + 39.84
    # sstfromocean3 = t03*4.2345 - 930.42
    # sstfromocean4 = t04*3.6489 - 771.44
    # sstfromocean6 = t06*47.3833 - 12751.71
    # sstfromocean7 = t07*25.4563 - 6737.08
    # to disable sst, enable temps below
    sstfromocean1 = t01
    sstfromocean2 = t02
    sstfromocean3 = t03
    sstfromocean4 = t04
    sstfromocean6 = t06
    sstfromocean7 = t07
    
    qah = ( np.average([sstfromocean1,sstfromocean2],weights=[warea,earea]) - np.average([sstfromocean3,sstfromocean4],weights=[narea,sarea]))
    qaw = (sstfromocean1-sstfromocean2)
    oc1 = (sstfromocean2-sstfromocean1) /M1 * (Cp*rho*h1)
    oc2 = (t05-sstfromocean2) /M2 * (Cp*rho*h2)
    
    if ahtsst:
        # sstfromocean1 = t01*0.9641 + 10.88
        # sstfromocean2 = t02*0.8688 + 39.84
        sstfromocean3 = t03*4.2345 - 930.42
        sstfromocean4 = t04*3.6489 - 771.44
        sstfromocean6 = t06*47.3833 - 12751.71
        sstfromocean7 = t07*25.4563 - 6737.08

    msenonwest = np.average( [mse(sstfromocean2),mse(sstfromocean3),mse(sstfromocean4)], weights = [earea,narea,sarea] )
    atm_div_w = mse_gamma_w*(mse(sstfromocean1) - msenonwest) 
    atm01 = atm_div_w + mse_int_w

    msenoneast = np.average( [mse(sstfromocean1),mse(sstfromocean3),mse(sstfromocean4)], weights = [warea,narea,sarea] )
    atm_div_e = mse_gamma_e*(mse(sstfromocean2) - msenoneast)
    atm02 = atm_div_e + mse_int_e

    tropmean = (atm01 + atm02)/2
    msenonnorth = np.average( [mse(sstfromocean1),mse(sstfromocean2),mse(t06)], weights = [warea,earea,hnarea] )
    # msenonnorth = mse(t06)
    atm_div_n = mse_gamma_n*(mse(sstfromocean3) - msenonnorth) #+ mse_gamma_n_trop*(mse(sstfromocean3) - tropmean)
    atm03 = atm_div_n + mse_int_n

    msenonsouth = np.average( [mse(sstfromocean1),mse(sstfromocean2),mse(t07)], weights = [warea,earea,hsarea] )
    # msenonsouth = mse(t07)
    atm_div_s = mse_gamma_s*(mse(sstfromocean4) - msenonsouth) #+ mse_gamma_s_trop*(mse(sstfromocean4) - tropmean)
    atm04 = atm_div_s + mse_int_s

    sstfromocean1 = t01
    sstfromocean2 = t02
    sstfromocean3 = t03
    sstfromocean4 = t04
    sstfromocean6 = t06
    sstfromocean7 = t07
    # set up linear system of equations for Ah, Aw and epsilon
    # syntax: [X1 * Aw, X2 * Ah] 
    epsilon = 0.48
    o1 = [qaw*(1-epsilon)*oc1, qah*(1-epsilon)*oc1, 1]
    o2 = [qaw*oc2, qah*oc2, 1]
    o3 = [qaw*(epsilon*(sstfromocean2-sstfromocean3)+(1-epsilon)*(sstfromocean1-sstfromocean3))/M3 * (Cp*rho*h3)*M3/(M3+M4), qah*(epsilon*(sstfromocean2-sstfromocean3)+(1-epsilon)*(sstfromocean1-sstfromocean3))/M3 * (Cp*rho*h3)*M3/(M3+M4), 1]
    o4 = [qaw*(epsilon*(sstfromocean2-sstfromocean4)+(1-epsilon)*(sstfromocean1-sstfromocean4))/M4 * (Cp*rho*h4)*M4/(M3+M4), qah*(epsilon*(sstfromocean2-sstfromocean4)+(1-epsilon)*(sstfromocean1-sstfromocean4))/M4 * (Cp*rho*h4)*M4/(M3+M4), 1]
    targets = [ -37.05497791847675, -66.01889841098343 ,-2.609530888506832,-7.0966641085897555] # negative of SHF from ERA5
    # targets = [ shfall[i][0], shfall[i][1], shfall[i][2], shfall[i][3] ] # CMIP PI SHF, already negative so don't need to invert sign
    # targets = [-(swlw1 + atm01), -(swlw2 + atm02), -(swlw3 + atm03), -(swlw4 + atm04)]
    A = [o1,o2,o3,o4]
    Aw,Ah,oc_int = np.linalg.lstsq(A,targets,rcond=None)[0]
    
    # PRESCRIBING ERA5 VALUES
    # print("USING PRESCRIBED Ah,Aw,oc_int FROM ERA5")
    # Aw=2341446.5112397578 
    # Ah=5250625.344206382 
    # oc_int=-22.441997378133237

    # then readjust MSE intercept to balance all energy budgets
    ot01 = np.dot( o1, [Aw,Ah,oc_int])
    ot02 = np.dot( o2, [Aw,Ah,oc_int])
    ot03 = np.dot( o3, [Aw,Ah,oc_int])
    ot04 = np.dot( o4, [Aw,Ah,oc_int])

    mse_int_w = -( swlw1 + atm_div_w + ot01 )
    mse_int_e = -( swlw2 + atm_div_e + ot02 )
    mse_int_n = -( swlw3 + atm_div_n + ot03 )
    mse_int_s = -( swlw4 + atm_div_s + ot04 )
    
    if highlatcmipoht: #prescribe dOHT with cmip values, units W/m2
        ot6 = -oht0[i][4] #doht[i][4][0] #  W/m2  These are positive upwards, so take negative
        ot7 = -oht0[i][5] #doht[i][5][0]
    else:
        ot6=0
        ot7=0
    
    if ahtsst:
        sstfromocean6 = t06*47.3833 - 12751.71
        sstfromocean7 = t07*25.4563 - 6737.08
    
    atm_div_hn = mse_gamma_hn*(mse(sstfromocean6) - mse(sstfromocean3))
    mse_int_hn = -(swlw6 + atm_div_hn + ot6)

    atm_div_hs = mse_gamma_hs*(mse(sstfromocean7) - mse(sstfromocean4)) 
    mse_int_hs = -(swlw7 + atm_div_hs + ot7)
    
    return mse_int_w, mse_int_e, mse_int_n, mse_int_s, mse_int_hn, mse_int_hs, Aw,Ah,oc_int

#####################################################################


#%% MAIN LOOP

        
def model( fb_iter , itr, c, t01,t02,t03,t04,t05,t06,t07,
          mse_gamma_w,mse_gamma_e,mse_gamma_n,mse_gamma_s,
          mse_gamma_hn,mse_gamma_hs,mse_int_w,mse_int_e,
          mse_int_n,mse_int_s,swlw1,swlw2,swlw3,swlw4,swlw6,swlw7): # for a given fb set
    
    i = fb_iter
    
    #define initial conditions for each run
    
    if cmip6params:
        t01 = t0all[i][0]
        t02 = t0all[i][1]
        t03 = t0all[i][2]
        t04 = t0all[i][3]
        t05 = np.average( [t0all[i][2],t0all[i][3]],weights=[narea,sarea] ) # weighted average of boxes 3&4
        t06 = t0all[i][4]
        t07 = t0all[i][5]

#         mse_gamma_w = mse_gamma_all[i][0]
#         mse_gamma_e = mse_gamma_all[i][1]
#         mse_gamma_n = mse_gamma_all[i][2]
#         mse_gamma_s = mse_gamma_all[i][3]
#         mse_gamma_hn = mse_gamma_all[i][4]
#         mse_gamma_hs = mse_gamma_all[i][5]

#         mse_int_w = mse_int_all[i][0]
#         mse_int_e = mse_int_all[i][1]
#         mse_int_n = mse_int_all[i][2]
#         mse_int_s = mse_int_all[i][3]

        # swlw1 = swlwall[i][0]
        # swlw2 = swlwall[i][1]
        # swlw3 = swlwall[i][2]
        # swlw4 = swlwall[i][3]
        # swlw6 = swlwall[i][4]
        # swlw7 = swlwall[i][5]
    
        

    # use tuning function to set params based on initial conditions above
    mse_int_w,mse_int_e,mse_int_n,mse_int_s,mse_int_hn, mse_int_hs,Aw,Ah,oc_int = atmocntuning(t01,t02,t03,t04,t05,t06,t07, 
                                                                        mse_gamma_w,mse_gamma_e,mse_gamma_n,mse_gamma_s,
                                                                        mse_gamma_hn,mse_gamma_hs,mse_int_w,mse_int_e,
                                                                        mse_int_n,mse_int_s,swlw1,swlw2,swlw3,swlw4,swlw6,swlw7,i)
    
    # if i==0:
    print(f"i={i} Aw={Aw} Ah={Ah} oc_int={oc_int}")
        
    ### change epsilon AFTER initial energy balancing. else, OHT+AHT will always balance and cancel out any epsilon change ###
    epsilon = 0.48
    
    #debugging
    # if i==0:
    # print("mse_int_w=",mse_int_w)
    # print("mse_int_e=",mse_int_e)
    # print("mse_int_n=",mse_int_n)
    # print("mse_int_s=",mse_int_s)
    # print("mse_int_hs=",mse_int_hs)
    # print("mse_int_hn=",mse_int_hn)
    
    if fb_adapt_bool:
        print(f"{i+1} of {len(fb)} fb_adapt ON")
    elif usefullfb:
        print(f"{i+1} of {len(fb)} full fb time series")
    elif usetoaanoms:
        print(f"{i+1} of {len(fb)} full TOA time series")
    else:
        print(f"{(i+1)*(itr+1)*(c+1) } of {len(fb)*len(co2)*changenum} fb={fb[i]}")
       
    ###############################
    #### CALC INIT OHT AND AHT ####
    ###############################
    
    # get SST estimates from ocean temps. from ERA5 notebook on casper
    # sstfromocean1 = t01*0.9641 + 10.88
    # sstfromocean2 = t02*0.8688 + 39.84
    # sstfromocean3 = t03*4.2345 - 930.42
    # sstfromocean4 = t04*3.6489 - 771.44
    # sstfromocean6 = t06*47.3833 - 12751.71
    # sstfromocean7 = t07*25.4563 - 6737.08
    
    sstfromocean1 = t01
    sstfromocean2 = t02
    sstfromocean3 = t03
    sstfromocean4 = t04
    sstfromocean6 = t06
    sstfromocean7 = t07
    
    q=Ah*( np.average([sstfromocean1,sstfromocean2],weights=[warea,earea]) - np.average([sstfromocean3,sstfromocean4],weights=[narea,sarea])) + Aw*(sstfromocean1-sstfromocean2)  #for north and south
    
    if i==0:
        print(f"q0 = {q}")
       
    q4 = q*M4/(M3+M4)
    q3 = q-q4
    # east feeding west with water not lost to ekman
    ocean1 = q*(1-epsilon)*(sstfromocean2-sstfromocean1) + oc_int*M1/(Cp*rho*h1) #convert intercept W/m2 -> T*m3/s
    #undercurrent (T5) feeding into east (T2)
    ocean2 = q*(t05-sstfromocean2) + oc_int*M2/(Cp*rho*h2)
    #east and west feeding north via ekman
    ocean3 = q3*epsilon*(sstfromocean2-sstfromocean3) + q3*(1-epsilon)*(sstfromocean1-sstfromocean3) + oc_int*M3/(Cp*rho*h3)
    #east and west feeding south via ekman
    ocean4 = q4*epsilon*(sstfromocean2-sstfromocean4) + q4*(1-epsilon)*(sstfromocean1-sstfromocean4) + oc_int*M4/(Cp*rho*h4)
    #undercurrent being fed by north and south, as average of two weighted by volumes of each
    ocean5 = q3*(sstfromocean3-t05) + q4*(sstfromocean4-t05) #np.average( [q*(T3-T5), q2*(T4-T5)], weights=[M3,M4] ) 
    
    #get ocean heat transport in W/m2
    o1wm2 = ocean1/M1 * (Cp*rho*h1)
    o2wm2 = ocean2/M2 * (Cp*rho*h2)
    o3wm2 = ocean3/M3 * (Cp*rho*h3)
    o4wm2 = ocean4/M4 * (Cp*rho*h4)
    o5wm2 = ocean5/M5 * (Cp*rho*underdepth)
    
    
    
    # use sst for AHT calc only
    if ahtsst:
        # sstfromocean1 = t01*0.9641 + 10.88
        # sstfromocean2 = t02*0.8688 + 39.84
        sstfromocean3 = t03*4.2345 - 930.42
        sstfromocean4 = t04*3.6489 - 771.44
        sstfromocean6 = t06*47.3833 - 12751.71
        sstfromocean7 = t07*25.4563 - 6737.08
    
    msenonwest = np.average( [mse(sstfromocean2),mse(sstfromocean3),mse(sstfromocean4)], weights = [earea,narea,sarea] )
    atm_div_w = mse_gamma_w*(mse(sstfromocean1) - msenonwest) + mse_int_w
    
    msenoneast = np.average( [mse(sstfromocean1),mse(sstfromocean3),mse(sstfromocean4)], weights = [warea,narea,sarea] )
    atm_div_e = mse_gamma_e*(mse(sstfromocean2) - msenoneast) + mse_int_e
    
    # tropmean = (atm_div_w + atm_div_e)/2
    msenonnorth = np.average( [mse(sstfromocean1),mse(sstfromocean2),mse(sstfromocean6)], weights = [warea,earea,hnarea] )
    # msenonnorth = mse(sstfromocean6)
    atm_div_n = mse_gamma_n*(mse(sstfromocean3) - msenonnorth) + mse_int_n # + mse_gamma_n_trop*(mse(sstfromocean3) - tropmean)
    
    msenonsouth = np.average( [mse(sstfromocean1),mse(sstfromocean2),mse(sstfromocean7)], weights = [warea,earea,hsarea] )
    # msenonsouth = mse(sstfromocean7)
    atm_div_s = mse_gamma_s*(mse(sstfromocean4) - msenonsouth) + mse_int_s # + mse_gamma_s_trop*(mse(sstfromocean4) - tropmean)
    
    atm_div_hn = mse_gamma_hn*(mse(sstfromocean6) - mse(sstfromocean3)) + mse_int_hn
    
    atm_div_hs = mse_gamma_hs*(mse(sstfromocean7) - mse(sstfromocean4)) + mse_int_hs
    
    
   # Calculate initial global mean AHT
    meanAHT0 = np.average( [atm_div_w,atm_div_e,atm_div_n,atm_div_s,atm_div_hn,atm_div_hs], weights=[warea,earea,narea,sarea,hnarea,hsarea] ) 
    
    #################################################
    ############## Initialize arrays ################
    #################################################
    
    t1=[t01]
    t2=[t02]
    t3=[t03]
    t4=[t04]
    t5=[t05]
    t6=[t06]
    t7=[t07]
    R1=[]; R2=[]; R3=[]; R4=[]
    div=[[atm_div_w],[atm_div_e],[atm_div_n],[atm_div_s],[atm_div_hn],[atm_div_hs]]
    circ=[[],[],[],[]]
    if highlatcmipoht:
        ot=[[o1wm2],[o2wm2],[o3wm2],[o4wm2],[o5wm2],[doht[i][4][0]],[doht[i][5][0]]]
    else:
        ot=[[o1wm2],[o2wm2],[o3wm2],[o4wm2],[o5wm2],[0],[0]]
    tempfb=[[],[],[],[],[],[]]
    mAHT=[]

    ##################################################################################
    #################### TIME INTEGRATION ############################################
    ##################################################################################
    
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

        # get SST estimates from ocean temps. from ERA5 notebook on casper
        # sstfromocean1 = T1*0.9641 + 10.88
        # sstfromocean2 = T2*0.8688 + 39.84
        # sstfromocean3 = T3*4.2345 - 930.42
        # sstfromocean4 = T4*3.6489 - 771.44
        # sstfromocean6 = T6*47.3833 - 12751.71
        # sstfromocean7 = T7*25.4563 - 6737.08
        
        sstfromocean1 = T1
        sstfromocean2 = T2
        sstfromocean3 = T3
        sstfromocean4 = T4
        sstfromocean6 = T6
        sstfromocean7 = T7
        
        #ocean transport m3/s from tropics to extropics
        if oht_fixed:
            q=Ah*( np.average([t01,t02],weights=[warea,earea]) - np.average([t03,t04],weights=[narea,sarea])) + Aw*(t01-t02)
        else:
            q=Ah*( np.average([sstfromocean1,sstfromocean2],weights=[warea,earea]) - np.average([sstfromocean3,sstfromocean4],weights=[narea,sarea])) + Aw*(sstfromocean1-sstfromocean2)  
            
        #for north and south
       
        q4 = q*M4/(M3+M4)
        q3 = q-q4
        
        # "global" mean of box model
        Tanommean = 1/np.sum(areas)*( (T1-t01)*areas[0] + (T2-t02)*areas[1] + (T3-t03)*areas[2] + (T4-t04)*areas[3] + (T6-t06)*areas[4] + (T7-t07)*areas[5] )
        # use linear relationship to go from box global to estimate of real global, as defined in boxmodel.ipynb
        globalTS = Tanommean * 1.4685590563197095 + 1.1226523565765463 # this is an estimate of global temp anom, can use for feedbacks. got rid of intercept to make it start at 0 
        
       
        # get SST estimates from ocean temps. from ERA5 notebook on casper
        # if ahtsst:
            # sstfromocean1 = T1*0.9641 + 10.88
            # sstfromocean2 = T2*0.8688 + 39.84
            # sstfromocean3 = T3*4.2345 - 930.42
            # sstfromocean4 = T4*3.6489 - 771.44
            # sstfromocean6 = T6*47.3833 - 12751.71
            # sstfromocean7 = T7*25.4563 - 6737.08
        
        ### calculate initial tropical mean divAHT ###
        ahttropmean0 = (atm_div_w + atm_div_e)/2
        
    ###############################################################
    ## BOX 1 ### WEST #############################################
    ###############################################################
        #fb param in west only depends on local temp
        
        if fb_adapt_bool:
            fbparam11 = fbadapt[i][0][0]
            fbparam12 = fbadapt[i][0][1]
            fbparam13 = fbadapt[i][0][2]
            # fb1 = (T1 - t01) * fbadapt[i][0][0]
            fb1 = fbparam11 * globalTS + fbparam12 * ( (T1-t01) - (T2-t02) ) + fbparam13 # global def
            
        elif usefullfb:
            fb1 = globalTS * fb[i,0,t] #fb[i][0][t]
        elif usetoaanoms:
            fb1 = fb[i][0][t] 
        else:
            # fb1 = (T1 - t01) * fb[i][0] # local temp def
            fb1 = globalTS * fb[i][0] # global temp def
            
            # debugging
            # if t==0:
            #     print(f"fb1 = {fb[i][0]}")

        # atm_div = mse_gamma*(mse_mean - mse(T1)) + mse_int 
        if aht_fixed:
            msenonwest = np.average( [mse(t02),mse(t03),mse(t04)], weights = [earea,narea,sarea] )
            atm_div_w = mse_gamma_w*(mse(t01) - msenonwest) + mse_int_w
        elif lowlatcmipaht:
            atm_div_w = cmipaht[i][0][t]
        else:
            msenonwest = np.average( [mse(sstfromocean2),mse(sstfromocean3),mse(sstfromocean4)], weights = [earea,narea,sarea] )
            atm_div_w = mse_gamma_w*(mse(sstfromocean1) - msenonwest) + mse_int_w
            
        # if (i==0) and (t==0):
        #     print(f"atm_div_w={atm_div_w}")
        
        if regionalforcing:
            F = 6.35 #7.05 #6.2 #4.7
            # F = forcings[i][0]
        else:
            F = co2[c]
            
        if usetoaanoms:
            F = 0
        
        R= swlw1 + atm_div_w + F + fb1 #+ nino[0][0][t]  #use t01 for fixed OLR
        
        if ((t+1)%500==0): #write out data every 1/12 of a year
            tempfb[0].append( fb1+F )
            div[0].append(atm_div_w)
            circ[0].append(q*(1-epsilon)*(T2-T1))
        H1= 1/(Cp*rho*h1) * R
        R1.append(R)

        
    ###################################################################
    #### BOX 2 ### EAST ############################################### 
    ###################################################################        
        
        if fb_adapt_bool: #single regional feedback parameter
            fbparam21 = fbadapt[i][1][0]
            fbparam22 = fbadapt[i][1][1]
            fbparam23 = fbadapt[i][1][2]
            # fb2 = fbparam21 * (T2 - t02) + fbparam22 * ( (T1-t01) - (T2-t02) ) #+ fbparam23
            fb2 = fbparam21 * globalTS + fbparam22 * ( (T1-t01) - (T2-t02) ) + fbparam23 # global def
            #local feedback: (Tnow - T0) * lambda
             
        elif usefullfb:
            fb2 = globalTS * fb[i,1,t] #fb[i][1][t]
        elif usetoaanoms:
            fb2 = fb[i][1][t]
        # elif uselinearfb:
        #     fb2 = globalTS * linearfb[i,0,t]
        else:
            # fb2 = (T2 - t02) * fb[i][1] # local
            fb2 = globalTS * fb[i][1] # global
            
        # atm_div = mse_gamma*(mse_mean - mse(T2)) + mse_int #old version gamma*(Tmean-T2)
        if aht_fixed:
            msenoneast = np.average( [mse(t01),mse(t03),mse(t04)], weights = [warea,narea,sarea] )
            atm_div_e = mse_gamma_e*(mse(t02) - msenoneast) + mse_int_e
        elif lowlatcmipaht:
            atm_div_e = cmipaht[i][1][t]
        else:
            msenoneast = np.average( [mse(sstfromocean1),mse(sstfromocean3),mse(sstfromocean4)], weights = [warea,narea,sarea] )
            atm_div_e = mse_gamma_e*(mse(sstfromocean2) - msenoneast) + mse_int_e
        # if (i==0) and (t==0):
        #     print(f"atm_div_e={atm_div_e}")
        
        if regionalforcing:
            F = 7.83 #8.5 #8.6
            # F = forcings[i][1]
        else:
            F = co2[c]
        
        if usetoaanoms:
            F = 0
            
        R= swlw2 + atm_div_e + F + fb2 
        
        if ((t+1)%500==0): #write out data every 1/12 of a year
            tempfb[1].append( fb2+F )
            
            div[1].append(atm_div_e)
            circ[1].append(q*(T4-T2))
        H2= 1/(Cp*rho*h2) * R
        R2.append(R)

    
    ### calculate tropical mean divAHT anom ###
        ahttropmeananom = (atm_div_w + atm_div_e)/2 - ahttropmean0
        ahttropmean = (atm_div_w + atm_div_e)/2
    ########################################################################
    #### BOX 3 ### NORTH ###################################################
    ########################################################################
    
        if fb_adapt_bool:
            fbparam31 = fbadapt[i][2][0]
            fbparam32 = fbadapt[i][2][1]
            fbparam33 = fbadapt[i][2][2]
            # fb3 = fbparam31 * (T3 - t03) + fbparam32 * ((T1-t01) - (T3-t03)) #+ fbparam33
            # fb3 = fbparam31 * globalTS + fbparam32 * ((T1-t01) - (T3-t03)) + fbparam33
            fb3 = fbparam31 * globalTS + fbparam32 * ((T1-t01) - (T2-t02)) + fbparam33
            
        elif usefullfb:
            fb3 = globalTS * fb[i,2,t] #fb[i][2][t]
        elif usetoaanoms:
            fb3 = fb[i][2][t]
        elif uselinearfb:
            fb3 = globalTS * linearfb[i,1,t]
        else:
            # fb3 = (T3 - t03) * fb[i][2] # local
            fb3 = globalTS * fb[i][2] # global
            
        # atm_div = mse_gamma*(mse_mean - mse(T3)) + mse_int 
        if aht_fixed:
            msenonnorth = np.average( [mse(t01),mse(t02),mse(t06)], weights = [warea,earea,hnarea] )
            atm_div_n = mse_gamma_n*(mse(t03) - msenonnorth) + mse_int_n
        elif lowlatcmipaht:
            atm_div_n = cmipaht[i][2][t]
        else:
            msenonnorth = np.average( [mse(sstfromocean1),mse(sstfromocean2),mse(sstfromocean6)], weights = [warea,earea,hnarea] )
            # msenonnorth = np.mean( [mse(sstfromocean1),mse(sstfromocean2),mse(sstfromocean6)] )
            # msenonnorth = mse(sstfromocean6)
            atm_div_n = mse_gamma_n*(mse(sstfromocean3) - msenonnorth) + mse_int_n
            
        # subtract tropical mean divAHT anom
        # ahttropmeananom_n = ahttropmeananom * ( narea/(narea+sarea) ) # take area weighted amount
        # atm_div_n -= ahttropmeananom_n
        
        # add tropical mean term
        # atm_div_n += mse_gamma_n_trop*( mse(sstfromocean3) - ahttropmean )
        
        # if (i==0) and (t==0):
        #     print(f"atm_div_n={atm_div_n}")
        
        if regionalforcing:
            F = 6.35 #7.4 #8.1
            # F = forcings[i][2]
        else:
            F = co2[c]
            
        if usetoaanoms:
            F = 0
        
        R= swlw3 + atm_div_n + F + fb3 #+ nino[2][0][t]
        
        if ((t+1)%500==0): #write out data every 1/12 of a year
            tempfb[2].append( fb3+F )
            
            div[2].append(atm_div_n)
            circ[2].append(q*epsilon*(T2-T3) + q*(1-epsilon)*(T1-T3))
        H3= 1/(Cp*rho*h3) * R
        R3.append(R)

    
    ########################################################################
    #### BOX 4 ### SOUTH ###################################################
    ########################################################################
    
        if fb_adapt_bool:
            fbparam41 = fbadapt[i][3][0]
            fbparam42 = fbadapt[i][3][1]
            fbparam43 = fbadapt[i][3][2]
            # fb4 = fbparam41 * (T4 - t04) + fbparam42 * ((T1-t01) - (T4-t04)) #+ fbparam43
            # fb4 = fbparam41 * globalTS + fbparam42 * ((T1-t01) - (T4-t04)) + fbparam43 # global def
            fb4 = fbparam41 * globalTS + fbparam42 * ((sstfromocean1-t01) - (T2-t02)) + fbparam43 # global def
    
        elif usefullfb:
            fb4 = globalTS * fb[i,3,t] #fb[i][3][t]  
        elif uselinearfb:
            fb4 = globalTS * linearfb[i,2,t]
        elif usetoaanoms:
            fb4 = fb[i][3][t] 
        else:
            # fb4 = (T4 - t04) * fb[i][3]
            fb4 = globalTS * fb[i][3]
            
        if aht_fixed:
            msenonsouth = np.average( [mse(t01),mse(t02),mse(t07)], weights = [warea,earea,hsarea] )
            atm_div_s = mse_gamma_s*(mse(t04) - msenonsouth) + mse_int_s
        elif lowlatcmipaht:
            atm_div_s = cmipaht[i][3][t]
        else:
            msenonsouth = np.average( [mse(sstfromocean1),mse(sstfromocean2),mse(sstfromocean7)], weights = [warea,earea,hsarea] )
            # msenonsouth = np.mean( [mse(sstfromocean1),mse(sstfromocean2),mse(sstfromocean7)] )
            # msenonsouth = mse(sstfromocean7)
            atm_div_s = mse_gamma_s*(mse(sstfromocean4) - msenonsouth) + mse_int_s
            
        # subtract tropical mean divAHT anom
        # ahttropmeananom_s = ahttropmeananom * ( sarea/(narea+sarea) ) # take area weighted amount
        # atm_div_s -= ahttropmeananom_s

        # add tropical mean term
        # atm_div_s += mse_gamma_s_trop*( mse(sstfromocean4) - ahttropmean )
        
        # if (i==0) and (t==0):
        #     print(f"atm_div_s={atm_div_s}")
        
        if regionalforcing:
            F = 7.08 #8.43 #8.7
            # F = forcings[i][3]
        else:
            F = co2[c]
            
        if usetoaanoms:
            F = 0
            
        R= swlw4 + atm_div_s + F + fb4 #+ aaht #+ nino[2][0][t]
        
        if ((t+1)%500==0): #write out data every 1/12 of a year
            tempfb[3].append( fb4+F )
            
            div[3].append(atm_div_s)
            circ[3].append(q*epsilon*(T2-T3) + q*(1-epsilon)*(T1-T3))
        H4= 1/(Cp*rho*h4) * R
        R4.append(R)

    
    
    ########################################################################
    #### BOX 6 ### HIGH LATITUDE NORTH #####################################
    ########################################################################
    
        
        # fb6 = (T6 - t06) * fb[i][4] # local temp
        if usefullfb:
            fb6 = globalTS * fb[i,4,t] #fb[i][4][t]
        elif usetoaanoms:
            fb6 = fb[i][4][t]
        else:
            fb6 = globalTS * (-1.2) # global temp 
#         if fb_adapt_bool:
       
#             # try lambda1 * dt_local + lambda2 * (dt_west - dt_local)
#             fbparam61 = fbadapt[i][4][0]
#             fbparam62 = fbadapt[i][4][1]
#             fbparam63 = fbadapt[i][4][2]
#             # fb6 = fbparam61 * (T6 - t06) + fbparam62 * ((T1-t01) - (T6-t06)) #+ fbparam43
#             fb6 = fbparam61 * globalTS 
        

        if aht_fixed:
            atm_div_hn = mse_gamma_hn*(mse(t06) - mse(t03)) + mse_int_hn
        elif highlatcmipaht:
            atm_div_hn = cmipaht[i][4][t]
        else:
            atm_div_hn = mse_gamma_hn*(mse(sstfromocean6) - mse(sstfromocean3)) + mse_int_hn
        
        if regionalforcing:
            F = 3.00 #3.69 #4.2
            # F = forcings[i][4]
        else:
            F = co2[c]
            
        if usetoaanoms:
            F = 0
        
        R= swlw6 + atm_div_hn + F + fb6 
        
        if ((t+1)%500==0): #write out data every 1/12 of a year
            tempfb[4].append( fb6+F )
        
            div[4].append(atm_div_hn)
        
        H6= 1/(Cp*rho*h6) * R

    
    ########################################################################
    #### BOX 7 ### HIGH LATITUDE SOUTH #####################################
    ########################################################################
    
        if usefullfb:
            fb7 = globalTS * fb[i,5,t] #fb[i][5][t] 
        elif usetoaanoms:
            fb7 = fb[i][5][t]
        else:
        # fb7 = (T7 - t07) * fb[i][5] # local temp
            fb7 = globalTS * (-1.2) # global temp 
#         if fb_adapt_bool:
            
#             # try lambda1 * dt_local + lambda2 * (dt_west - dt_local)
#             fbparam71 = fbadapt[i][5][0]
#             fbparam72 = fbadapt[i][5][1]
#             fbparam73 = fbadapt[i][5][2]
#             # fb7 = fbparam71 * (T7 - t07) + fbparam72 * ((T1-t01) - (T7-t07)) #+ fbparam43
#             fb7 = fbparam71 * globalTS 
        

        if aht_fixed:
            atm_div_hs = mse_gamma_hs*(mse(t07) - mse(t04)) + mse_int_hs
        elif highlatcmipaht:
            atm_div_hs = cmipaht[i][5][t]
        else:
            atm_div_hs = mse_gamma_hs*(mse(sstfromocean7) - mse(sstfromocean4)) + mse_int_hs
        
        if regionalforcing:
            F = 4.04 #5.72 #6.2
            # F = forcings[i][5]
        else:
            F = co2[c]
            
        if usetoaanoms:
            F = 0
        
        R= swlw7 + atm_div_hs + F + fb7 
        
        if ((t+1)%500==0): #write out data every 1/12 of a year
            tempfb[5].append( fb7+F )
        
            div[5].append(atm_div_hs)
    
       
        H7= 1/(Cp*rho*h7) * R
        
    
    ########################################################################
    ########## Solve ocean transport equations #############################
    ########################################################################
        
        sstfromocean1 = T1
        sstfromocean2 = T2
        sstfromocean3 = T3
        sstfromocean4 = T4
        sstfromocean6 = T6
        sstfromocean7 = T7
        
        #ocean transport, units T*m3/s
        
        if oht_fixed:
            # east feeding west with water not lost to ekman
            ocean1 = q*(1-epsilon)*(t02-t01) + oc_int*M1/(Cp*rho*h1) #convert intercept W/m2 -> T*m3/s
            #undercurrent (T5) feeding into east (T2)
            ocean2 = q*(t05-t02) + oc_int*M2/(Cp*rho*h2)
            #east and west feeding north via ekman
            ocean3 = q3*epsilon*(t02-t03) + q3*(1-epsilon)*(t01-t03) + oc_int*M3/(Cp*rho*h3) #- 6*M3/(Cp*rho*h3) #CESM adjustment
            #east and west feeding south via ekman
            ocean4 = q4*epsilon*(t02-t04) + q4*(1-epsilon)*(t01-t04) + oc_int*M4/(Cp*rho*h4) #- 2*M3/(Cp*rho*h3)
            #undercurrent being fed by north and south, as average of two weighted by volumes of each
            ocean5 = q3*(t03-t05) + q4*(t04-t05) #np.average( [q*(T3-T5), q2*(T4-T5)], weights=[M3,M4] ) 
        else:
            # east feeding west with water not lost to ekman
            ocean1 = q*(1-epsilon)*(sstfromocean2-sstfromocean1) + oc_int*M1/(Cp*rho*h1) #convert intercept W/m2 -> T*m3/s
            # if t==0:
            #     print(f"run {i} int term={oc_int}")
            #undercurrent (T5) feeding into east (T2)
            ocean2 = q*(T5-sstfromocean2) + oc_int*M2/(Cp*rho*h2)
            #east and west feeding north via ekman
            ocean3 = q3*epsilon*(sstfromocean2-sstfromocean3) + q3*(1-epsilon)*(sstfromocean1-sstfromocean3) + oc_int*M3/(Cp*rho*h3) #- 6*M3/(Cp*rho*h3) #CESM adjustment
            # if t==0:
            #     print(f"box3 esp term={q3*epsilon*(T2-T3) + q3*(1-epsilon)*(sstfromocean1-T3)} int term={oc_int*M3/(Cp*rho*h3)}")
            #east and west feeding south via ekman
            ocean4 = q4*epsilon*(sstfromocean2-sstfromocean4) + q4*(1-epsilon)*(sstfromocean1-sstfromocean4) + oc_int*M4/(Cp*rho*h4) #- 2*M3/(Cp*rho*h3)
            # if t==0:
            #     print(f"box4 esp term={q4*epsilon*(T2-T4) + q4*(1-epsilon)*(T1-T4)} int term={oc_int*M4/(Cp*rho*h4)}")
            #undercurrent being fed by north and south, as average of two weighted by volumes of each
            ocean5 = q3*(sstfromocean3-T5) + q4*(sstfromocean4-T5) #np.average( [q*(T3-T5), q2*(T4-T5)], weights=[M3,M4] ) 
        
        if lowlatcmipoht: #prescribe dOHT with cmip values, units W/m2. add anomaly value to PI value
            ocean1 = (doht[i][0][t] + (-oht0[i][0]))/(Cp*rho*h1) * M1 # convert from W/m2 to T*m3/s. take negative of PI value, as is defined positive up 
            ocean2 = (doht[i][1][t] + (-oht0[i][1]))/(Cp*rho*h2) * M2
            ocean3 = (doht[i][2][t] + (-oht0[i][2]))/(Cp*rho*h3) * M3 # convert from W/m2 to T*m3/s. take negative of PI value, as is defined positive up 
            ocean4 = (doht[i][3][t] + (-oht0[i][3]))/(Cp*rho*h4) * M4

        if highlatcmipoht: #prescribe dOHT with cmip values, units W/m2. add anomaly value to PI value
            ocean6 = (doht[i][4][t] + (-oht0[i][4]))/(Cp*rho*h6) * M6 # convert from W/m2 to T*m3/s. take negative of PI value, as is defined positive up 
            ocean7 = (doht[i][5][t] + (-oht0[i][5]))/(Cp*rho*h7) * M7

    ##########################################################################
    ########################## AHT anom ######################################
    ##########################################################################
    
        # weighted mean AHT: should be 0? ERA mean is +1.79
        # if subtract net anomaly from everywhere, this assumes that everywhere is contributing equally
        if submeanAHT:
            meanAHTanom = np.average( [atm_div_w,atm_div_e,atm_div_n,atm_div_s,atm_div_hn,atm_div_hs], weights=[warea,earea,narea,sarea,hnarea,hsarea] ) - meanAHT0
        else:
            meanAHTanom = 0
        # convert from W/m2 to T/s by dividing by (Cp*rho*hX)

    ##########################################################################
    ################### Solve temperature evolution equations ################
    ##########################################################################
    
        #temperature evolution equations
        T1=T1 + dt/M1 * (M1*H1 + ocean1 - M1*(meanAHTanom/(Cp*rho*h1)) ) 
        T2=T2 + dt/M2 * (M2*H2 + ocean2 - M2*(meanAHTanom/(Cp*rho*h2)) ) #+ nino[1][0][t] * dt/(3600*24*30)
        T3=T3 + dt/M3 * (M3*H3 + ocean3 - M3*(meanAHTanom/(Cp*rho*h3)) ) #+ (-0.04*dt/(3600*24*30)) #add cooling trend from upwelling of .02 deg/month
        T4=T4 + dt/M4 * (M4*H4 + ocean4 - M4*(meanAHTanom/(Cp*rho*h4)) ) #+ (-0.04*dt/(3600*24*30))
        T5=T5 + dt/M5 * ocean5 

        if highlatcmipoht:
            T6=T6 + dt/M6 * (M6*H6 + ocean6 - M6*(meanAHTanom/(Cp*rho*h6)) ) # High lat NH, assuming no ocean connection
            T7=T7 + dt/M7 * (M7*H7 + ocean7 - M7*(meanAHTanom/(Cp*rho*h7)) ) # High lat SH, assuming no ocean connection
        else:
            T6=T6 + dt*(H6 - (meanAHTanom/(Cp*rho*h6)) ) # High lat NH, assuming no ocean connection
            T7=T7 + dt*(H7 - (meanAHTanom/(Cp*rho*h7)) ) # High lat SH, assuming no ocean connection
            
    ##############################################################################
    ################# Convert ocean transport to W/m2 ############################
    ##############################################################################

        #get ocean heat transport in W/m2
        o1wm2 = ocean1/M1 * (Cp*rho*h1)
        o2wm2 = ocean2/M2 * (Cp*rho*h2)
        o3wm2 = ocean3/M3 * (Cp*rho*h3)
        o4wm2 = ocean4/M4 * (Cp*rho*h4)
        o5wm2 = ocean5/M5 * (Cp*rho*underdepth)
        if highlatcmipoht:
            o6wm2 = ocean6/M6 * (Cp*rho*h6)
            o7wm2 = ocean7/M7 * (Cp*rho*h7)
        else:
            o6wm2 = 0
            o7wm2 = 0
        
    ##############################################################################
    ################## Write out data ############################################
    ##############################################################################
    
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
            
            ot[5].append(o6wm2)
            ot[6].append(o7wm2)
            
            # mAHT.append(meanAHT)
            
        # if t==0:
            
            # mAHT.append(meanAHT)
    
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
    # allR.append([R1,R2,R3,R4])
    atmosdiv.append(np.asarray(div))
    # meanaht.append( np.asarray(mAHT) )
    
    return allT,ocean,atmosdiv,totfb
 

allT=[]
meanT=[]
allR=[]
ocean=[]
atmosdiv=[]
totfb=[]
meanaht=[]

changenum=1
