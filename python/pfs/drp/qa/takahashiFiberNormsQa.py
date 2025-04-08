import eups
from lsst.daf.butler import Butler
import pandas as pd
import numpy as np
#%matplotlib widget
import matplotlib

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clip
import time
import csv
import scipy.interpolate as interp
import math
from astropy.coordinates import SkyCoord
import sys
import glob, os
import time
from statistics import mean, median,variance,stdev
from astropy.time import Time
from scipy.signal import medfilt
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import matplotlib.gridspec as gridspec

from pfs.datamodel import FiberStatus, PfsConfig, TargetType, PfsFiberNorms
from pfs.utils.fiberids import FiberIds

from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
#from ..utils.math import robustRms
from datetime import datetime, timedelta, time
from matplotlib.colors import Normalize
pfspipe2d = eups.getSetupVersion("pfs_pipe2d")
datastore = '/work/datastore'
collections = 'PFS/calib/pipe2d-1675/run21/verifyCalib.uniformity.20250322a'#'PFS/calib/pipe2d-1675/run21/calib.20250328a', (old) 'PFS/calib/pipe2d-1675/run21/verifyCalib.uniformity.20250322a'
butler = Butler(datastore, collections=collections)
list_spec = list(np.arange(1,5,1))

calibdata = pd.read_csv('/home/ayumitk/run21/calibdata_run21.csv', header=0)
trace = calibdata[(calibdata.name=='Trace')|(calibdata.name=='Trace uniformity')]
print(len(trace))
trace = trace.drop_duplicates(subset='visit_start',keep='first') #exclude duplicated visit number in calibration data list
trace = trace.reset_index()
trace = trace.sort_values('visit_start',ignore_index=True)
print(len(trace))

trace_visit_start = np.array([int(v) for v in np.array(trace.visit_start)])
print(len(trace_visit_start))
trace['visit_start_int'] = trace_visit_start
trace = trace.reset_index()



#a = 'b'
#visit_list = ['121090','121095','121099','121161','121187','121198','121209','121213','121314','121335',
#              '121501','121959','122107','122427','122532','122656','122970','122973','123086']

a = 'r' #exclude '121209, 121335 because it's m-arm'

visit_list = ['121090','121095','121099','121161','121187','121198','121213','121314',
             '121501','121959','122107','122427','122532','122656','122970','122973','123086']

#a = 'm'
#visit_list = [121209,121335]

print(len(visit_list))


outPath=f'compare_quartz_result/{a}-arm'
resultfile = f'compquartz_run21_{a}.dat'
28
19
19
17
vmin=0.97; vmax=1.03
vmin2=0.0; vmax2=0.05
wminlist = [450,650,950]
wmaxlist= [650,950,1250]
Measures Sigma
from pfs.utils.fiberids import FiberIds
fiberIds=FiberIds()
df_fibid = pd.DataFrame(fiberIds.data)


### Get MTP info. ####

fiberIds=FiberIds()
df_fibid = pd.DataFrame(fiberIds.data)

# routine to get MTP group
def get_mtpgroup(df):
    df['mtpGroup'] = df.mtp_A[0:6]
    return df





n_visit = len(visit_list)
leng_array = n_visit

#photon noise per spectrograph
noise= np.zeros((leng_array,4))


def func_sigma_irq(Q3,Q1,data,coff):
    q3, q1 = np.nanpercentile(data, [Q3,Q1], axis=1)
    sigma_iqr_perfib=coff*(q3 - q1)

    q3, q1 = np.nanpercentile(data, [Q3,Q1])
    sigma_iqr_flat = coff*(q3 - q1)

    var_true= sigma_iqr_flat**2 - var3

    if var_true < 0:
       sigma_true =0.
    else:
        sigma_true = np.sqrt(var_true)

    return sigma_iqr_perfib, sigma_iqr_flat, sigma_true





if a == 'b':
    wmin=wminlist[0]; wmax=wmaxlist[0]
elif (a == 'r') or ( a == 'm'):
    wmin=wminlist[1]; wmax=wmaxlist[1]
elif a == 'n':
    wmin=wminlist[2]; wmax=wmaxlist[2]




if not os.path.exists(outPath):
    os.makedirs(outPath)




#sigma measured from standard deviation
list_sigma_std = np.zeros(leng_array)

#---sigma measured from IQR
#sigma of quartz ratio per spectrograph (include photon noise)
list_sigma_obs = np.zeros((leng_array,4,4))# --(n_visit,n_spec,n_iqr_case)

#sigma of quartz ratio per spectrograph (subtracted photon noise)
list_sigma_tru = np.zeros((leng_array,4,4))



#sigma in all fibers
#n_spec = 4 #<------ a number of operated spectrogrph (i.e., In run19, operated spectrographs are SM2,3,4)
list_sigma_fib = np.full((leng_array,600,4),np.nan) #-----frame: (visit number, number of fibers each spectrograph, number of spectrographs)
df_tmp = pd.DataFrame(columns=['fiberId','sigma'])


# save estimated value
with open (resultfile, 'w') as fcomp:
    print(f'visit0,visit,hst,insrot,azimuth,sigma_std_median,sigma_std_95,2sigma_std,2sigma_std_95,sigma_iqr', file=fcomp)



for i,v in enumerate(visit_list):

    v1 = 121314#int(visit_list[0]) #<------------------Reference visit
    v2 = int(v)

    df_mtp=df_fibid[['fiberId', 'mtp_A']]
    df_mtp=df_mtp.apply(get_mtpgroup, axis=1, result_type='expand')
    df_mtp['arm'] = a
    df_mtp['SM'] = int(0)
    df_mtp['sigma'] = np.nan


    #try:

    arr = np.zeros((1))

    for q, s in enumerate(list_spec):
        s = s.item()


        try:
            dataId1 = dict(visit=v1, spectrograph=s, arm=a)#<------------------Reference visit
            dataId2 = dict(visit=v2, spectrograph=s, arm=a)#<------------------Target visit

            pfsArm1=butler.get("pfsArm", dataId1)#<------------------Reference visit

            pfsArm2=butler.get("pfsArm", dataId2)

            pfsConfig = butler.get("pfsConfig", dataId2)
            pfsConfig = pfsConfig.select(targetType=~TargetType.ENGINEERING)
            pfsConfig = pfsConfig.select(fiberStatus = 1)

            pfsArm1 = pfsArm1[np.isin(pfsArm1.fiberId,pfsConfig.fiberId)]
            pfsArm2 = pfsArm2[np.isin(pfsArm2.fiberId,pfsConfig.fiberId)]


            df_mtp.loc[df_mtp['fiberId'].isin(pfsArm2.fiberId),'SM'] = int(s)

        except:
            print(v2)


        try:
            ### This block will be removed due to overlap with above block ###
            dataId1 = dict(visit=v1, spectrograph=s, arm=a)#<------------------Reference visit
            dataId2 = dict(visit=v2, spectrograph=s, arm=a)#<------------------Target visit

            pfsArm1=butler.get("pfsArm", dataId1)#<------------------Reference visit

            pfsArm2=butler.get("pfsArm", dataId2)

            pfsConfig = butler.get("pfsConfig", dataId2)
            pfsConfig = pfsConfig.select(targetType=~TargetType.ENGINEERING)
            pfsConfig = pfsConfig.select(fiberStatus = 1)
            pfsArm1 = pfsArm1[np.isin(pfsArm1.fiberId,pfsConfig.fiberId)]
            pfsArm2 = pfsArm2[np.isin(pfsArm2.fiberId,pfsConfig.fiberId)]

            df_mtp.loc[df_mtp['fiberId'].isin(pfsArm2.fiberId),'SM'] = int(s)

            ##################################################################



            ratio = np.nan
            ratio = pfsArm2.flux/pfsArm1.flux
            # discard around the edge of wavelength (noisy)
            ratio = np.where((pfsArm2.wavelength>wmin)&(pfsArm2.wavelength<wmax),ratio,np.nan)
            ratio = np.where(np.isinf(ratio),np.nan,ratio)

            xx, yy = np.meshgrid(np.nanmedian(ratio, axis=0), np.nanmedian(ratio, axis=1)/np.nanmedian(ratio))
            ratio = ratio/xx/yy

            #### Consider photon noise for each arm
            flux1 = np.nanmedian(pfsArm1.flux)
            flux2 = np.nanmedian(pfsArm2.flux)
            var1 = np.nanmedian(pfsArm1.covar[0][0])
            var2 = np.nanmedian(pfsArm2.covar[0][0])
            var3 = ((flux2/flux1)**2)*((var2/(flux2**2))+(var1/(flux1**2)))
            noise[i][q] = np.sqrt(var3)


            # Calculate standard deviation for both raw data
            sigma=np.nanstd(ratio, axis=1)
            arr = np.append(arr, sigma)



            # Calculate IQR_sigma for both raw data
            ## Q99-Q1
            c = 0.215
            q3,q1 = 99,1
            sigma_iqr1=func_sigma_irq(q3,q1,ratio,c)[0]
            list_sigma_obs[i,q,0] = func_sigma_irq(q3,q1,ratio,c)[1]
            list_sigma_tru[i,q,0] = func_sigma_irq(q3,q1,ratio,c)[2]

            ## Q90-Q10
            c = 0.39
            q3,q1 = 90,10
            sigma_iqr2=func_sigma_irq(q3,q1,ratio,c)[0]
            list_sigma_obs[i,q,1] = func_sigma_irq(q3,q1,ratio,c)[1]
            list_sigma_tru[i,q,1] = func_sigma_irq(q3,q1,ratio,c)[2]

            ## Q84-Q16
            c = 0.50
            q3,q1 = 84,16
            sigma_iqr3=func_sigma_irq(q3,q1,ratio,c)[0]
            list_sigma_obs[i,q,2] = func_sigma_irq(q3,q1,ratio,c)[1]
            list_sigma_tru[i,q,2] = func_sigma_irq(q3,q1,ratio,c)[2]

            ## Q75-Q25
            c = 0.741
            q3,q1 = 75,25
            sigma_iqr4=func_sigma_irq(q3,q1,ratio,c)[0]
            list_sigma_obs[i,q,3] = func_sigma_irq(q3,q1,ratio,c)[1]
            list_sigma_tru[i,q,3] = func_sigma_irq(q3,q1,ratio,c)[2]



        except LookupError:
            sigma_iqr4 = np.full(len(pfsConfig.fiberId),np.nan)



        df_mtp.loc[df_mtp['fiberId'].isin(pfsArm2.fiberId),['sigma']]  = sigma_iqr4


        ################## Observing parameters #############

        ### For target visit
        inr = pfsArm2.metadata['INSROT']
        azi = pfsArm2.metadata['AZIMUTH']
        hst = pfsArm2.metadata['HST']



    arr = arr[1:]



    use = arr

    list_sigma_std[i] = np.nanmedian(use)


    df_mtp.to_csv(f'{outPath}/sigma_perFiber_perMTP_{v2}over{v1}.csv',index=False)

    robustSigma = np.nanmedian(df_mtp['sigma'])
    with open (resultfile, 'a') as fcomp:
            print(f'{v1},{v2},{hst},{inr},{azi},{np.median(arr):.4f},{np.nanpercentile(arr, 95):.4f},{np.median(arr*2):.4f},{np.nanpercentile(arr*2, 95):.4f},{robustSigma}',file=fcomp)

######## Define a function to take quartz ratio


if a == 'b':
    wmin=wminlist[0]; wmax=wmaxlist[0]
elif (a == 'r') or ( a == 'm'):
    wmin=wminlist[1]; wmax=wmaxlist[1]
elif a == 'n':
    wmin=wminlist[2]; wmax=wmaxlist[2]


def func_quartzratio(v0,v,a,s):
    #dataId.update(arm=a, spectrograph=s)
    arm0 = butler.get("pfsArm",dict(visit=v0, arm=a, spectrograph=s))
    arm = butler.get("pfsArm", dict(visit=v, arm=a, spectrograph=s))
    #arm0 = arm0[arm0.fiberId.isin(arm.fiberId)]
    pfsConfig = butler.get("pfsConfig", dict(visit=v, arm=a, spectrograph=s))


    # use only science fibers
    pfsConfig=pfsConfig[(pfsConfig.targetType!=TargetType.ENGINEERING)&(pfsConfig.spectrograph==s)]
    pfsConfig = pfsConfig.select(fiberStatus=1)

    arm0 = arm0[np.isin(arm0.fiberId,pfsConfig.fiberId)]
    arm = arm[np.isin(arm.fiberId,pfsConfig.fiberId)]

    quartz = arm.flux/arm0.flux

    # apply median filter
    quartz_mf = medfilt(quartz, kernel_size=(1,15))
    # discard around the edge of wavelength (noisy)
    quartz = np.where((arm.wavelength>wmin)&(arm.wavelength<wmax),quartz,np.nan)
    quartz = np.where(np.isinf(quartz),np.nan,quartz)

    # pick-up wavelength for median filter-ed spectra
    x_mf = arm.wavelength[~np.isnan(quartz)]
    quartz_mf = np.where((arm.wavelength>wmin)&(arm.wavelength<wmax),quartz_mf,np.nan)
    #ratio_mf = medfilt(ratio[~np.isnan(ratio)], kernel_size=(1,15))

    xx, yy = np.meshgrid(np.nanmedian(quartz, axis=0), np.nanmedian(quartz, axis=1)/np.nanmedian(quartz))
    quartz = quartz/xx/yy
    xx_mf, yy_mf = np.meshgrid(np.nanmedian(quartz_mf, axis=0), np.nanmedian(quartz_mf, axis=1)/np.nanmedian(quartz_mf))
    quartz_mf = quartz_mf/xx_mf/yy_mf
    #print(arm.fiberId[np.isnan(np.nanstd(ratio, axis=1))])
    #ptp = np.nanmax(ratio, axis=1) - np.nanmin(ratio, axis=1)
    fibs = np.array([np.full_like(arm.wavelength[np.where(arm.fiberId==f)[0][0]], f) for f in arm.fiberId])

    return arm, quartz, fibs, pfsConfig
### Convert UTC to HST #####
import pytz


def utc2hst(utc_str):

    # ISO to datetime
    utc_dt = datetime.fromisoformat(utc_str.replace('Z', '+00:00'))

    # set time zone
    utc_dt = pytz.utc.localize(utc_dt) if utc_dt.tzinfo is None else utc_dt

    # convert to the hst time zone（UTC-10:00）
    hst_tz = pytz.timezone('US/Hawaii')
    hst_dt = utc_dt.astimezone(hst_tz)

    hst_formatted = hst_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    hst_formatted = hst_formatted[:-5]

    return hst_formatted

for i,v in enumerate(visit_list[0:1]):

    v = int(v)
    v0 = 121314

    datalist = f'{outPath}/sigma_perFiber_perMTP_{v}over{v0}.csv'
    df = pd.read_csv(datalist,index_col=False)
    df =df[df.mtpGroup.str.contains('U')|df.mtpGroup.str.contains('D')]
    df = df[df.SM != 0]
    df = df.reset_index()
    result_median = pd.read_csv(resultfile)
    hst = np.array(result_median['hst'])[result_median.visit==v][0][0:6]
    ins =np.array( result_median['insrot'][result_median.visit==v])[0]
    azi = np.round(result_median['azimuth'][result_median.visit==v].values,2)[0]



    fig = plt.figure(num='fiberNormsQA', figsize=(30,55), clear=True, facecolor='w', edgecolor='k')

    fs = 20
    FS = fs+5

    ###### For extracting odd fiber ######
    if a =='b':
        sigma_flag_entire = 0.01090
    elif (a == 'r') | (a == 'n' ) | (a == 'm'):
        sigma_flag_entire = 0.00799

    arr = np.array(df.sigma)
    df_odd = df[(arr>sigma_flag_entire)] #extract odd MTPs/fibers
    df_odd = df_odd.sort_values('sigma', ascending=False)
    0
    if len(df_odd) > 10:
        df_odd = df_odd[0:15]
    else:
        df_odd = df_odd.head(int(len(df_odd)*0.25))

    df_odd = df_odd.reset_index()
    n_odd = len(df_odd)

    ncols_odd = 5
    nrows_odd = int(n_odd/ncols_odd) ### the number of rows for plot of odd fibers


    nrows = 4 + 4 # later four rows are for showing spectra

    gs_master = gridspec.GridSpec(nrows=nrows,ncols=6,height_ratios=np.full(nrows,1),wspace=0.7,hspace=0.5)


    ax5 = fig.add_subplot(gs_master[1,0], aspect='equal') #--For PFI image with pfsArm
    ax6 = fig.add_subplot(gs_master[1,1], aspect='equal') #--For PFI image with pfsArm
    ax7 = fig.add_subplot(gs_master[2,0], aspect='equal') #--For PFI image with pfsArm
    ax8 = fig.add_subplot(gs_master[2,1], aspect='equal') #--For PFI image with pfsArm
    ax9 = fig.add_subplot(gs_master[3,0], aspect='equal') #--For PFI image with fiberNorms
    ax10 = fig.add_subplot(gs_master[3,1])#, aspect='equal') #--For PFI image with fiberNorms

    ### 2D spectral image
    gs_spec1 = gridspec.GridSpecFromSubplotSpec(nrows = 2, ncols=2, subplot_spec= gs_master[0:1,0:1]) #--For 2D spectral image
    ax13 = fig.add_subplot(gs_master[1:3,2:5])
    ax13.set_xlabel('wavelength [nm]', fontsize=fs)
    ax13.set_ylabel('fiberId', fontsize=fs)


    ## Something
    gs_spec2 = gridspec.GridSpecFromSubplotSpec(nrows = 1, ncols=2, subplot_spec= gs_master[0:1,0:1])
    ax14 = fig.add_subplot(gs_master[3,2:5])

    ## MTP group vs. spectrograph ID
    gs_spec3 = gridspec.GridSpecFromSubplotSpec(nrows = 3, ncols=1, subplot_spec= gs_master[0:1,0:1]) #--For 2D spectral image
    ax15 = fig.add_subplot(gs_master[1:4,5])


    ############################################### PLOT DATA ###############################################################

    ############################### [5] ############################################

    ax14.set_title('[5] Sigma per fiber',fontsize=fs)
    ax14.scatter(df.mtpGroup,df.sigma,c='black')

    sigma_median = np.nanmedian(df.sigma)
    ax14.axhline(sigma_median,ls='dashed',c='blue',zorder=0)
    ax14.annotate('median',(0.9, 0.1),c='blue',fontsize=14,xycoords='axes fraction')
    clip = 4
    sigma_of_sigma = sigma_median +clip*np.std(df.sigma)
    ax14.axhline(sigma_of_sigma,ls='dashed',c='red',zorder=0)
    ax14.text('D2-1-4',1.2*sigma_of_sigma,'4 sigma',c='red',fontsize=14)
    ax14.text(0.85,0.9,f'${clip}\sigma$ = {sigma_of_sigma:.4f}',color='red',transform=ax14.transAxes,fontsize=fs-3,bbox=dict(facecolor='white', alpha=0.5))
    ax14.text(0.85,0.8,f'median = {sigma_median:.4f}',color='blue',transform=ax14.transAxes,fontsize=fs-3,bbox=dict(facecolor='white', alpha=0.5))

    ax14.grid(axis='both', which='both', linestyle='--', linewidth=0.5)
    ax14.set_xlabel('MTP group',fontsize=15)
    ax14.set_ylabel('$\sigma$ (0.741*(Q75-Q25))',fontsize=15)
    ax14.tick_params('x',labelrotation=90)

    ############################### [3] ############################################

    ax15.set_title('[3] Used MTPs',fontsize=fs)
    xdata,ydata = df.SM[df.SM!=0.],df.mtpGroup[df.SM!=0.]
    ax15.scatter(xdata,ydata,c='olivedrab')
    ## The following MTPs had not been estimated sigma due to some error (e.g., could not read butler)
    xdata,ydata = df.SM[df.SM==0.],df.mtpGroup[df.SM==0.]
    ax15.scatter(xdata,ydata,c='gray')

    ax15.grid(axis='both', which='both', linestyle='--', linewidth=0.5)
    ax15.set_xticks([0,1,2,3,4])
    ax15.set_xlabel('spectrograph ID', fontsize=fs)
    ax15.set_ylabel('MTP group', fontsize=fs-5)



    ############## [2] 2D spec of quartz ratio ###############
    #sigma_flag = np.nanmedian(df.sigma)+1*np.nanstd(df.sigma) # flag for odd fibers

    if a == 'b':
        wmin=wminlist[0]; wmax=wmaxlist[0]
    elif (a == 'r') or ( a == 'm'):
        wmin=wminlist[1]; wmax=wmaxlist[1]
    elif a == 'n':
        wmin=wminlist[2]; wmax=wmaxlist[2]

    for s in list_spec:

        arm = func_quartzratio(v0,v,a,s)[0]
        ratio = func_quartzratio(v0,v,a,s)[1]
        fibs = func_quartzratio(v0,v,a,s)[2]
        pfsconfig = func_quartzratio(v0,v,a,s)[3]

        # plot quartz ratio as contour (fiberId vs wavelength)
        sc6 = ax13.scatter(arm.wavelength, fibs, c=ratio,vmin=vmin, vmax=vmax, s=0.6, alpha=1.0, label='quartz')

        oddfib = df.fiberId[(df.SM==s)&(df.sigma>sigma_flag_entire)].values
        for i_fib in oddfib:
            ax13.scatter(wmin,i_fib,marker='+',color='red',s=150,alpha=0.8)



        ######################################## [1] PFI IMAGES (pfsArm.flux/pfsArm.norm) ########################################
        fig.text(0.15,0.78, '[1] Quartz ratio measured from pfsArm',fontsize=fs)
        n_sigma= 2
        sc0 = ax5.scatter(pfsconfig.pfiCenter[:,0],pfsconfig.pfiCenter[:,1], c = np.array(df.sigma[df['SM']==s])*n_sigma, vmin=vmin2, vmax=vmax2, s=30.0, alpha=1.0, label=f'{n_sigma}sigma')
        ax5.set_xlim(xmin=-250, xmax=250)
        ax5.set_ylim(ymin=-250, ymax=250)
        ax5.yaxis.set_ticks_position('left')
        ax5.set_title(f'{n_sigma}sigma', fontsize=fs)
        ax5.set_xlabel('X(PFI) [mm]', fontsize=fs)
        ax5.set_ylabel('Y(PFI) [mm]', fontsize=fs)

        #Plot median
        median_array = np.array([np.nanmedian(ratio[i_fiber])for i_fiber in range(len(ratio))])
        sc1=ax6.scatter(pfsconfig.pfiCenter[:,0], pfsconfig.pfiCenter[:,1], c=median_array,vmin=vmin, vmax=vmax,
                                         s=30.0, alpha=1.0, label=f'median per fiber')
        ax6.set_xlim(xmin=-250, xmax=250)
        ax6.set_ylim(ymin=-250, ymax=250)
        ax6.yaxis.set_ticks_position('left')
        ax6.set_title(f'median per fiber', fontsize=fs)
        ax6.set_xlabel('X(PFI) [mm]', fontsize=fs)
        ax6.set_ylabel('Y(PFI) [mm]', fontsize=fs)


        i_pixel = 1500
        ratio_lam1 = np.array([ratio[i_fiber][i_pixel]for i_fiber in range(len(ratio))])
        sc2=ax7.scatter(pfsconfig.pfiCenter[:,0], pfsconfig.pfiCenter[:,1], c=ratio_lam1, vmin=vmin, vmax=vmax,
                                     s=30.0, alpha=1.0)
        ax7.set_xlim(xmin=-250, xmax=250)
        ax7.set_ylim(ymin=-250, ymax=250)
        ax7.yaxis.set_ticks_position('left')
        lam_point = np.round(arm.wavelength[0][i_pixel],3)
        ax7.set_title(f'at {lam_point} [nm]', fontsize=fs)
        ax7.set_xlabel('X(PFI) [mm]', fontsize=fs)
        ax7.set_ylabel('Y(PFI) [mm]', fontsize=fs)


        i_pixel = 3500
        ratio_lam2 = np.array([ratio[i_fiber][i_pixel]for i_fiber in range(len(ratio))])
        sc3=ax8.scatter(pfsconfig.pfiCenter[:,0], pfsconfig.pfiCenter[:,1], c=ratio_lam2, vmin=vmin, vmax=vmax,
                                         s=30.0, alpha=1.0, label=f'median per fiber')
        ax8.set_xlim(xmin=-250, xmax=250)
        ax8.set_ylim(ymin=-250, ymax=250)
        ax8.yaxis.set_ticks_position('left')
        lam_point = np.round(arm.wavelength[0][i_pixel],3)
        ax8.set_title(f'at  {lam_point} [nm]', fontsize=fs)
        ax8.set_xlabel('X(PFI) [mm]', fontsize=fs)
        ax8.set_ylabel('Y(PFI) [mm]', fontsize=fs)


        ######################################## [4]PFI IMAGES (fiberNorms) ########################################
        fig.text(0.15,0.59, '[4] fiberNorms.values of target quartz',fontsize=fs)
        dataId = dict(visit=v, arm=a, spectrograph=s)
        pfsConfig = butler.get("pfsConfig", dataId)
        pfsConfig = pfsConfig.select(fiberStatus=1)

        fiberNorms=butler.get("fiberNorms", dict(visit=v, arm=a, spectrograph=s))
        fiberNorms = fiberNorms[np.isin(fiberNorms.fiberId,pfsConfig.fiberId)]
        fiberNorms.values = np.where((fiberNorms.wavelength>wmin)&(fiberNorms.wavelength<wmax),fiberNorms.values,np.nan)
        fiberNorms.plot(pfsConfig, axes=ax9, lower=2.5, upper=2.5)

        ax9.set_title(f'median per fiber', fontsize=fs)
        ax9.set_xlabel('X(PFI) [mm]', fontsize=fs)
        ax9.set_ylabel('Y(PFI) [mm]', fontsize=fs)

    fig.colorbar(sc0, ax=ax5, location='right', fraction=0.04, alpha=1.0)
    fig.colorbar(sc1, ax=ax6, location='right', fraction=0.04, alpha=1.0)
    fig.colorbar(sc2, ax=ax7, location='right', fraction=0.04, alpha=1.0)
    fig.colorbar(sc3, ax=ax8, location='right', fraction=0.04, alpha=1.0)

    #################### [4] 2D image of fiberNomrs.values #############

    indices = np.argsort(pfsConfig.fiberId)
    xx = pfsConfig.pfiCenter[indices, 0]
    yy = pfsConfig.pfiCenter[indices, 1]

    values = np.nanmedian(fiberNorms.values, axis=1)
    good = np.isfinite(values)
    median = np.median(values[good])
    rms = np.sqrt(np.mean(np.square(values[good])))#robustRms(values[good])
    lower = median - 2.5*np.std(values)#max(median - 2.5*rms, np.nanmin(values))
    upper = median + 2.5*np.std(values)#min(median + 2.5*rms, np.nanmax(values))
    norm = Normalize(vmin=lower, vmax=upper)
    fib_norm = np.array([np.full_like(fiberNorms.wavelength[np.where(fiberNorms.fiberId==f)[0][0]], f) for f in fiberNorms.fiberId])
    sc5 = ax10.scatter(fiberNorms.wavelength,fib_norm,c = fiberNorms.values,vmin=lower,vmax=upper,s=10,cmap='coolwarm')
    divider = make_axes_locatable(ax10) #axに紐付いたAxesDividerを取得
    cax = divider.append_axes("right", size="5%", pad=0.1) #append_axesで新しいaxesを作成

    ax10.set_ylabel('fiber index', fontsize=fs-3)
    ax10.set_xlabel('wavelength', fontsize=fs-3)
    ax10.set_title(f'2D spectrum', fontsize=fs)
    fig.colorbar(sc5, cax=cax)


    # ax13 - - [2] 2D spec of quartz ratio
    divider = make_axes_locatable(ax13) #axに紐付いたAxesDividerを取得
    cax = divider.append_axes("right", size="2%", pad=0.1) #append_axesで新しいaxesを作成

    ax13.set_xlabel('wavelength [nm]', fontsize=FS)
    ax13.set_ylabel('fiberId', fontsize=FS)
    ax13.set_title(f'[2] 2D spectrum of quartz ratio measured from pfsArm.flux',fontsize=fs)

    fig.colorbar(sc6, cax=cax)



    #################### [6] 1D spectra measured from pfsArm #############

    if len(df_odd) >= 15:
        gs_spec = gridspec.GridSpecFromSubplotSpec(nrows=nrows_odd, ncols=ncols_odd, subplot_spec=gs_master[4:7,:],wspace=0.3,hspace=0.5)
        fig.text(0.35,0.49, f'[6] Example spectra for flagged fibers with large flux scatters (red marked in [2])',fontsize=fs)
        i_row = 0
        i_col = 0
        for i_odd in range(n_odd):
            axs = fig.add_subplot(gs_spec[i_row,i_col])
            arm = df_odd['arm'].loc[i_odd]
            spec = df_odd['SM'].loc[i_odd]

            dataId0 = dict(visit=v0, spectrograph=spec, arm=arm)#<------------------Reference visit
            dataId = dict(visit=v, spectrograph=spec, arm=arm)#<------------------Target visit

            pfsArm0=butler.get("pfsArm", dataId0) # Reference visit
            pfsArm=butler.get("pfsArm", dataId) # Target visit

            flux0 = pfsArm0.flux/pfsArm0.norm
            flux = pfsArm.flux/pfsArm.norm
            ratio = flux/flux0

            fib = df_odd.fiberId[i_odd]
            inx = np.where(pfsArm.fiberId==fib)[0]
            n_sigma = 2


            # apply median filter
            ratio_mf = medfilt(ratio, kernel_size=(1,15))
            # discard around the edge of wavelength (noisy)
            ratio = np.where((pfsArm.wavelength>wmin)&(pfsArm.wavelength<wmax),ratio,np.nan)
            ratio = np.where(np.isinf(ratio),np.nan,ratio)

            # pick-up wavelength for median filter-ed spectra
            x_mf = pfsArm.wavelength[~np.isnan(ratio)]
            ratio_mf = np.where((pfsArm.wavelength>wmin)&(pfsArm.wavelength<wmax),ratio_mf,np.nan)
            #ratio_mf = medfilt(ratio[~np.isnan(ratio)], kernel_size=(1,15))

            xx, yy = np.meshgrid(np.nanmedian(ratio, axis=0), np.nanmedian(ratio, axis=1)/np.nanmedian(ratio))
            ratio = ratio/xx/yy
            xx_mf, yy_mf = np.meshgrid(np.nanmedian(ratio_mf, axis=0), np.nanmedian(ratio_mf, axis=1)/np.nanmedian(ratio_mf))
            ratio_mf = ratio_mf/xx_mf/yy_mf


            xdata = pfsArm.wavelength[inx]
            ydata = ratio[inx]
            axs.scatter(xdata,ydata, color='royalblue', s=10,label=f'{fib}')

            xdata_m = pfsArm.wavelength[inx]
            ydata_m = ratio_mf[inx]
            std = np.nanstd(ydata_m)
            axs.scatter(xdata_m,ydata_m, color='limegreen', s=10,label=f'{fib}(mf)')

            axs.text(0.68,0.9,f'$\sigma$ ='+ f'{df_odd.sigma.iloc[i_odd]:.4f}',transform=axs.transAxes,fontsize=fs-3,bbox=dict(facecolor='yellow', alpha=1.))
            #axs.axhline(y=np.nanmedian(ydata_m)+n_sigma*std, color='tomato', linestyle='dashed')
            #axs.axhline(y=np.nanmedian(ydata_m)-n_sigma*std, color='tomato', linestyle='dashed')
            axs.axhline(y=np.nanmedian(ydata_m)+n_sigma*df_odd.sigma.iloc[i_odd], color='tomato', linestyle='dashed')
            axs.axhline(y=np.nanmedian(ydata_m)-n_sigma*df_odd.sigma.iloc[i_odd], color='tomato', linestyle='dashed')

            axs.set(xlabel='wavelength ($\mathrm{\AA}$)',ylabel='normalized flux') # 全てのサブプロットに対してラベルを設定
            #axs.label_outer()  # 外側のサブプロットのみに軸ラベルを表示するようにする

            axs.set_ylim(ymin=np.nanmedian(ydata_m)-(n_sigma+6)*std, ymax=np.nanmedian(ydata_m)+(n_sigma+6)*std)
            #axs.set_ylim(ymin=0.6,ymax=1.4)
            axs.legend(loc = 'upper left',fontsize=fs-5)
            axs.grid(axis='y', which='both', linestyle='--', linewidth=0.5)

            axs.set_title(f'{df_odd.mtp_A[i_odd]}',fontsize=fs)

            if i_col != ncols_odd-1:
                i_col += 1
            elif i_col == ncols_odd-1:
                i_col = 0
                i_row += 1

        #################### [7] 1D spectra measured from fiberNorms #############

        gs_spec2 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=ncols_odd, subplot_spec=gs_master[7:,:],wspace=0.3,hspace=0.5)
        fig.text(0.35,0.19, f'[7] Randomly selected spectra obtained by fiberNorms.values',fontsize=fs)
        d = list(df.fiberId)
        list_ex_fibernorms = random.sample(d,ncols_odd) # extract fiberIds for plotting extrample spectra
        i_row = 0
        i_col = 0
        while i_col < ncols_odd:
            axs2 = fig.add_subplot(gs_spec2[i_row,i_col])
            dataId = dict(visit=v, spectrograph=spec, arm=arm)#<------------------Target visit
            fiberNorms=butler.get("fiberNorms", dataId) # Target visit
             # discard around the edge of wavelength (noisy)
            fiberNorms.values = np.where((fiberNorms.wavelength>wmin)&(fiberNorms.wavelength<wmax),fiberNorms.values,np.nan)
            fiberNorms.values = np.where(np.isinf(fiberNorms.values),np.nan,fiberNorms.values)

            fib_nrm = list_ex_fibernorms[i_col]
            inx_nrm = np.where(fiberNorms.fiberId==fib_nrm)[0]
            axs2.scatter(fiberNorms.wavelength[inx_nrm],fiberNorms.values[inx_nrm], color='navy', s=10,label=f'{fib_nrm}')

            xdata_m = fiberNorms.wavelength[inx_nrm]
            xdata_m = xdata_m[~np.isnan(fiberNorms.values[inx_nrm])]
            ydata_m = medfilt(fiberNorms.values[inx_nrm][~np.isnan(fiberNorms.values[inx_nrm])], kernel_size=15)


            ydata_m_ = medfilt(fiberNorms.values[inx_nrm][~np.isnan(fiberNorms.values[inx_nrm])], kernel_size=15)

            std = np.nanstd(ydata_m)
            axs2.scatter(xdata_m,ydata_m, color='limegreen', s=10,label=f'{fib_nrm}(mf)')

            axs2.axhline(y=np.nanmedian(ydata_m)+n_sigma*std, color='red', linestyle='dashed')
            axs2.axhline(y=np.nanmedian(ydata_m)-n_sigma*std, color='red', linestyle='dashed')

            axs2.set(xlabel='wavelength ($\mathrm{\AA}$)',ylabel='normalized flux')
            axs2.set_ylim(ymin=np.nanmedian(ydata_m)-(n_sigma+2)*std, ymax=np.nanmedian(ydata_m)+(n_sigma+2)*std)
            axs2.legend(loc = 'upper left',fontsize=fs-5)
            axs2.grid(axis='y', which='both', linestyle='--', linewidth=0.5)

            axs2.set_title(f'{df.mtp_A[df.fiberId==fib_nrm].to_string(index=False)}',fontsize=fs)
            i_col += 1




        ########################################################################################################### end plotting ##################################################################################################

        sigma_typ = np.nanmedian(df.sigma)


        if sigma_typ >= sigma_flag_entire:
            flag = 'Yes'
            strcolor = 'red'
        else:
            flag = 'No'
            strcolor = 'black'


    #### Obtain observation date

    obsdate = utc2hst(fiberNorms.metadata['DATEOBS'])


    plt.gcf().text(0.1,0.85,f'fiberNormsQA ver. 1.0', fontsize=fs-5)
    plt.gcf().text(0.7,0.85,f'Flag for fiberthrouput variation={flag}',color= strcolor, fontsize=FS,bbox=dict(facecolor='yellow', alpha=1.))
    plt.gcf().text(0.1,0.83,f'visit_target={v}', fontweight='bold',fontsize=FS)
    plt.gcf().text(0.1,0.82,f'visit_reference={v0}, arm={a}, obsdate={obsdate} ,insrot={ins}deg, azimuth={azi}deg',fontsize=FS)
    plt.gcf().text(0.1,0.81, f'datastore={datastore}, collections={collections}, pfs_pipe2d={pfspipe2d}',fontsize=FS)


    if (v==v0)  | (len(df_odd) == 0):

        plt.annotate(f'fiberNormsQA is to monitor fiber throughput variation. We took quartz flux ratios for some of the figures to investigate how much quartz flux\n varies with time.'+
                     'i.e., quartz ratio = pfsArm.flux(visit_target)/pfsArm.flux(visit_reference). '+
                     'Referenced quartz is the first one in the data set \n basically (see visit number above).\n'+
                     'Sigma is measured from 0.741*(Q75-Q25). '+'Descriptions for each sub-fig component are as follows.\n'+

                    '[1] PFI image of quartz ratios. 2 sigma per fier and median flux ratio per fiber are represent in the upper left and in the upper right figure, \n respectively.'+
                    ' We also represent median flux ratios at two different wavelength point via PFI images.\n'+
                    '[2] 2D spectrun of quartz ratio. If measure sigma of a fiber is larger then'+f' {sigma_flag_entire}, red closs marks fiber Id in the left side.\n'+
                    '[3] Used MTPs compared with corresponding spectrograph ID. The MTP groups in which all MTPs were used were represented in green, \n and partially unused MTPs are represented in grey.\n'+
                    '[4] fiberNorms.values of target quartz. Plotting bounds are '+r'2.5$\sigma$'+'\n'+
                    '[5] Measured sigma for each fibers. The median values in all MTP groups represent in blue dashed line and 4 sigma \nrepresent in red dashed line. This figure is to check flux variation with MTP unit. '+
                    'MTP groups above the red dashed line, \n indicating that the MTP group has a large flux variation. \n'
                    ,
                     (-95,-1.5), fontsize=fs+5, xycoords='axes fraction')


    else:

        plt.annotate(f'fiberNormsQA is to monitor fiber throughput variation. We took quartz flux ratios for some of the figures to investigate how much quartz flux\n varies with time.'+
                     'i.e., quartz ratio = pfsArm.flux(visit_target)/pfsArm.flux(visit_reference). '+
                     'Referenced quartz is the first one in the data set \n basically (see visit number above).\n'+
                     'Sigma is measured from 0.741*(Q75-Q25). '+'Descriptions for each sub-fig component are as follows.\n'+

                    '[1] PFI image of quartz ratios. 2 sigma per fier and median flux ratio per fiber are represent in the upper left and in the upper right figure, \n respectively.'+
                    ' We also represent median flux ratios at two different wavelength point via PFI images.\n'+
                    '[2] 2D spectrun of quartz ratio. If measure sigma of a fiber is larger then'+f' {sigma_flag_entire}, red closs marks fiber Id in the left side.\n'+
                    '[3] Used MTPs compared with corresponding spectrograph ID. The MTP groups in which all MTPs were used were represented in green.\n'+
                    '[4] fiberNorms.values of target quartz. Plotting bounds are '+ r'2.5$\sigma$'+'\n'+
                    '[5] Measured sigma for each fibers. The median values in all MTP groups represent in blue dashed line and 4 sigma \nrepresent in red dashed line. This figure is to check flux variation with MTP unit. '+
                    'MTP groups above the red dashed line, \n indicating that the MTP group has a large flux variation. \n'+
                    '[6] Spectra of FIBER with large sigma with a maximum of 15. Blue plots represent normalized spectra obtained by quartz ratio \n (i.e. pfsArm.flux(target)/pfsArm.norm(target)/pfsArm.flux(reference)/pfsArm.norm(reference)) and light green plots represent median filtered \n spectra.'+
                    '\nFiberId are shown in upper left. Mesured sigma per fiber are shown in upper right. Red dashed lines represent 2sigma lines of median filtered spectra.\n'+
                    '[7] Randomly selected spectra obtained from fiberNorms.values of target quartz',
                     (-5.5,-2.5), fontsize=fs+5, xycoords='axes fraction')

    outputfigurePath = f'output/fiberNormsQA/{a}-arm'
    if not os.path.exists(outputfigurePath):
        os.makedirs(outputfigurePath)

    fig.savefig(f'{outputfigurePath}'+f'/{v}_over{v0}.png',bbox_inches='tight')
    fig.tight_layout()
    fig.show()
