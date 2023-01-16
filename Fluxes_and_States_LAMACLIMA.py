# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:11:32 2022

@author: sdeherto
"""

#! /usr/bin/python

# This script is similar as the Fluxes_and_States_Masterscript from WAM-2layers from Ruud van der Ent, except that it threads atmospheric data at five pressure levels instead of model levels

# Includes a spline vertical interpolation for data on pressure levels from the EC-Earth climate model (Hazeleger et al., 2010)
# In this case the atmospheric data is provided on the following five pressure levels: 850 hPa, 700 hPa, 500 hPa, 300 hPa, 200 hPa 
# Includes a linear interpolation of the moisture fluxes over time (in def getrefined_new)
# We have implemented a datelist function so the model can run for multiple years without having problems with leap years

# My input data are monthly files with 3-hourly (surface variables) and 6-hourly data (atmospheric variables)

#%% Import libraries
import numpy as np
from netCDF4 import Dataset
import scipy.io as sio
from scipy import interpolate
from scipy.interpolate import interp1d
#import calendar
import os
import calendar
import sys

### check to make this compatible for all esms (drop cesm from name)
from getconstants_pressure_LAMACLIMA import getconstants_pressure_CESM

#from timeit import default_timer as timer
#import matplotlib.pyplot as plt    
import datetime as dt
import calendar
#import sys

# to create datelist
def get_times_daily(startdate, enddate): 
    """ generate a dictionary with date/times"""
    numdays = enddate - startdate
    dateList = []
    for x in range (0, numdays.days + 1):
        dateList.append(startdate + dt.timedelta(days = x))
    return dateList
         
#%%BEGIN OF INPUT (FILL THIS IN)

#when running this script in parallel you can use the 4 lines indicated below
#start_month = int(sys.argv[1])
#start_year = int(sys.argv[2])
#end_year = start_year + 1
#end_month = start_month + 1

# GET INFO FROM INPUT

model=sys.argv[1]
case=sys.argv[2]

#this values have to be filled depending in the new period of time. Now the test period corresponds to 15 day 
# Since 19/08/0024 to 2/09/0024
start_month = int(sys.argv[4])
start_year = int(sys.argv[3])
if model =='cesm':
    count_time = 4 # number of indices to get data from (4 timesteps a day, 6-hourly data)
else:
    count_time=8
end_year = start_year+1 
end_month = 12
start_day = 1

     
months = np.arange(start_month,end_month+1) #the list including the month at the end 
months_length_leap = [31,29,31,30,31,30,31,31,30,31,30,31]
months_length_nonleap = [31,28,31,30,31,30,31,31,30,31,30,31] # CESM does not have leap years
end_day = months_length_nonleap[end_month-1]
years = [start_year]#np.arange(start_year,end_year) #This value should be a list with the years within the period 

#datelist = get_times_daily(dt.date(years[0],months[0],1), dt.date(years[-1],months[-1], end_day))
            
def remove_leap_days(datelist):
    for jos in datelist:
        if ((jos.year % 400 == 0) or (jos.year % 100 != 0) and (jos.year % 4 == 0)):
            if ((jos.month==2) and (jos.day==29)):
                datelist.remove(jos)
    return datelist

# create datelist
if model !='cesm':
    datelist = get_times_daily(dt.date(years[0],months[0],1), dt.date(years[-1],months[-1], end_day))
    datelist=remove_leap_days(datelist)
else:
# create datelist without the if statement of Missisipi as there are no leap years
    datelist = get_times_daily(dt.date(years[0],months[0],start_day), dt.date(years[-1],months[-1], end_day))


# divt & count_time
if model != 'cesm':
    divt = 12
else:
    divt = 24 # division of the timestep, 24 means a calculation timestep of 6/24 = 0.25 hours (15min) (numerical stability purposes) 15 min
# Manage the extent of your dataset (FILL THIS IN)
# Define the latitude and longitude cell numbers to consider and corresponding lakes that should be considered part of the land


##adapt to res each ESM!!!!
if model =='cesm':
    latnrs = np.arange(0,192) # minimal domain 
    lonnrs = np.arange(0,288) 
elif model=='ecearth':
    latnrs = np.arange(0,256) # minimal domain 
    lonnrs = np.arange(0,512) 
elif model=='mpiesm':
    latnrs = np.arange(0,96) # minimal domain 
    lonnrs = np.arange(0,192) 

isglobal = 1 # fill in 1 for global computations (i.e. Earth round), fill in 0 for a local domain with boundaries

#END OF INPUT
#%% Datapaths (FILL THIS IN)
if model =='cesm':
   lsm_data_CESM = 'landmask_cesm.nc' # insert landseamask here
   area_mask = 'gridarea.nc'
   if case=='ctl':
     interdata_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/wam2layer/output/cesm/ctl/'  # insert interdata folder here
     input_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/cesm/archive/CTL_LAMACLIMA.e211.B2000cmip6.f09_g17.control-i308/' # insert input folder here
   elif case=='crop':
     interdata_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/wam2layer/output/cesm/crop/'  # insert interdata folder here
     input_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/cesm/archive/CROP_LAMACLIMA.e211.B2000cmip6.f09_g17.crop-i308.CROP/' # insert input folder here
   elif case=='frst':
     interdata_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/wam2layer/output/cesm/frst/'  # insert interdata folder here
     input_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/cesm/archive/FRST_LAMACLIMA.e211.B2000cmip6.f09_g17.frst-i308/' # insert input folder here
   elif case=='irr':
     interdata_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/wam2layer/output/cesm/irr/'  # insert interdata folder here
     input_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/cesm/archive/R_IRRIG_LAMACLIMA.e211.B2000cmip6.f09_g17.irrig-i308/' # insert input folder here

elif model=='ecearth':
   lsm_data_CESM = 'landmask_ecearth.nc' # insert landseamask here
   area_mask = 'gridarea_ecearth.nc'
   if case=='ctl':
     interdata_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/wam2layer/output/ecearth/ctl/'  # insert interdat$
     input_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/ecearth/ctl/'
   elif case=='crop':
     interdata_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/wam2layer/output/ecearth/crop/'  # insert interdat$
     input_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/ecearth/crop/'
   elif case=='frst':
     interdata_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/wam2layer/output/ecearth/frst/'  # insert interdat$
     input_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/ecearth/frst/'
   elif case=='irr':
     interdata_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/wam2layer/output/ecearth/irr/'  # insert interdat$
     input_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/ecearth/irr/'
elif model =='mpiesm':
   lsm_data_CESM = 'landmask_mpiesm.nc' # insert landseamask here
   area_mask = 'gridarea_mpiesm.nc'
   if case=='ctl':
     interdata_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/wam2layer/output/mpiesm/ctl/'  # insert inter$
     input_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/mpiesm/ctl/'
   elif case=='crop':
     interdata_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/wam2layer/output/mpiesm/crop/'  # insert inter$
     input_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/mpiesm/crop/'
   elif case=='frst':
     interdata_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/wam2layer/output/mpiesm/frst/'  # insert inter$
     input_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/mpiesm/frst/'
   elif case=='irr':
     interdata_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/wam2layer/output/mpiesm/irr/'  # insert inter$
     input_folder = r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/mpiesm/irr/'
#name_of_run = 'CTL'
#codigo_run = 'control-i308.cam.h3'
# other scripts use exactly this sequence, do not change it unless you change it also in the scripts

def data_path(yearnumber,month,a,model):   
# for all variables atmospheric and surface. This has to be adapted to each scenario 
    #t_f_data = os.path.join(input_folder, name_of_run + '_LAMACLIMA.e211.B2000cmip6.f09_g17.' + codigo_run +  str(yearnumber).zfill(4) + '-' +  str(month).zfill(2)+ '-'+ str(a).zfill(2) +'.nc')       

    if model=='cesm':
        q_f_data = os.path.join(input_folder, 'Q_last_30yrs_cesm.nc') #0specific humidity #0
        u_f_data = os.path.join(input_folder, 'U_last_30yrs_cesm.nc') #1
        v_f_data = os.path.join(input_folder, 'V_last_30yrs_cesm.nc') #2
        q2m_surface_data = os.path.join(input_folder, 'QREFHT_last_30yrs_cesm.nc') #3
        u10_surface_data = os.path.join(input_folder, 'UBOT_last_30yrs_cesm.nc') #4
        v10_surface_data = os.path.join(input_folder, 'VBOT_last_30yrs_cesm.nc') #5
        evaporation_data = os.path.join(input_folder, 'LHFLX_last_30yrs_cesm.nc') # evaporation #6
        precipitation_data = os.path.join(input_folder, 'PRECT_last_30yrs_cesm.nc') #  precipitation #7
        sp_data = os.path.join(input_folder, 'PS_last_30yrs_cesm.nc') # surface pressure data #8
    else:
        q_f_data = os.path.join(input_folder, 'hus_3hr.nc') #0specific humidity #0
        u_f_data = os.path.join(input_folder, 'ua_3hr.nc') #1
        v_f_data = os.path.join(input_folder, 'va_3hr.nc') #2
        q2m_surface_data = os.path.join(input_folder, 'huss_3hr.nc') #3
        u10_surface_data = os.path.join(input_folder, 'uas_3hr.nc') #4
        v10_surface_data = os.path.join(input_folder, 'vas_3hr.nc') #5
        evaporation_data = os.path.join(input_folder, 'hfls_3hr.nc') # evaporation #6
        precipitation_data = os.path.join(input_folder, 'pr_3hr.nc') #  precipitation #7
        sp_data = os.path.join(input_folder, 'ps_3hr.nc') # surface pressure data #8


    save_path = os.path.join(interdata_folder, str(yearnumber).zfill(4) + '-' + str(month).zfill(2) + '-' + str(a).zfill(2) + 'fluxes_storages.mat') #9
    return q_f_data,u_f_data,v_f_data,q2m_surface_data,u10_surface_data,v10_surface_data,evaporation_data,precipitation_data,sp_data, save_path  
 

#%% Code (no need to look at this for running)
if model=='cesm':
	var_list=['Q','QREFHT','PS','U','UBOT','VBOT','V','LHFLX','PRECT']
else:
	var_list=['hus','huss','ps','ua','uas','vas','va','hfls','pr']
# Determine the fluxes and states 
# In this defintion the vertical spline interpolation is performed to determine the moisture fluxes for the two layers at each grid cell

def getWandFluxes(latnrs,lonnrs,a,yearnumber,begin_time,count_time,
    density_water,latitude,longitude,g,A_gridcell,model):
    print(begin_time)
    print(count_time)
    if model=='cesm':
        p_coord='lev'
        p_levels=32
        p_factor=1 ##convert to hPa (1 for CESM)
        flip_axis=-1
    elif model =='ecearth':
        p_coord='plev'
        p_levels=8
        p_factor=100
        flip_axis=1
    elif model=='mpiesm':
        p_coord=['ap','b']
        p_levels=47
        p_factor=100
        flip_axis=1
    # specific humidity atmospheric data is 6-hourly (06.00,12.00,18.00, 00.00)
    q = Dataset(datapath[0], mode = 'r').variables[var_list[0]][begin_time:(begin_time+count_time+1),::flip_axis,latnrs,lonnrs] #kg/kg
    time = Dataset(datapath[0], mode = 'r').variables['time'][begin_time:(begin_time+count_time+1):]
    if model !='mpiesm':
       q_levels = Dataset(datapath[0], mode = 'r').variables[p_coord][::flip_axis]/p_factor #hPa
    else:
       q_levels = (Dataset(datapath[0], mode = 'r').variables[p_coord[0]][::flip_axis]/p_factor+Dataset(datapath[0], mode = 'r').variables[p_coord[1]][::flip_axis]/p_factor*100000) #hPa
    # specific humidity surface data is 6-hourly (03.00,06.00,09.00,12.00,15.00,18.00,21.00,00.00) 
    q2m = Dataset(datapath[3], mode ='r').variables[var_list[1]][begin_time:(begin_time+count_time+1),latnrs,lonnrs] #kg/kg #:267,134:578
    #time_q2m = Dataset(datapath[0], mode = 'r').variables['time'][:]    
    # surface pressure is 6-hourly data (03.00,06.00,09.00,12.00,15.00,18.00,21.00,00.00) 
    sp = Dataset(datapath[8], mode = 'r').variables[var_list[2]][begin_time:(begin_time+count_time+1),latnrs,lonnrs] # [Pa] #:267,134:578
  
    # read the u-wind data
    u = Dataset(datapath[1], mode = 'r').variables[var_list[3]][begin_time:(begin_time+count_time+1),::flip_axis,latnrs,lonnrs] #m/s 
    if model !='mpiesm':
       u_levels = Dataset(datapath[1], mode = 'r').variables[p_coord][::flip_axis]/p_factor #hPa
    else:
       u_levels = (Dataset(datapath[1], mode = 'r').variables[p_coord[0]][::flip_axis]/p_factor+Dataset(datapath[1], mode = 'r').variables[p_coord[1]][::flip_axis]/p_factor*100000) #hPa

    # wind at 10m, 6-hourly data
    u10 = Dataset(datapath[4], mode = 'r').variables[var_list[4]][begin_time:(begin_time+count_time+1),latnrs,lonnrs] #m/s "Lowest model level zonal wind" 
    v10 = Dataset(datapath[5], mode = 'r').variables[var_list[5]][begin_time:(begin_time+count_time+1),latnrs,lonnrs] #m/s "Lowest model level meridional wind
    # read the v-wind data
    v = Dataset(datapath[2], mode = 'r').variables[var_list[6]][begin_time:(begin_time+count_time+1),::flip_axis,latnrs,lonnrs] #m/s
    if model !='mpiesm':
       v_levels = Dataset(datapath[2], mode = 'r').variables[p_coord][::flip_axis]/p_factor #hPa
    else:
       v_levels = (Dataset(datapath[2], mode = 'r').variables[p_coord[0]][::flip_axis]/p_factor+Dataset(datapath[2], mode = 'r').variables[p_coord[1]][::flip_axis]/p_factor*100000) #hPa
  
    print ('Data is loaded', dt.datetime.now().time() )
    time = [0,1,2,3,4]
    intervals_regular = 40 # from five pressure levels the vertical data is interpolated to 40 levels
    ##Imme: location of boundary is hereby hard defined at model level 47 which corresponds with about 
    P_boundary = 0.72878581 * sp + 7438.803223  
    print(P_boundary)
    print(q_levels)
    p_low = q_levels[p_levels-1]*100 #Pa
    print('p_low: ',p_low)

 
    dp = (sp-p_low) /(intervals_regular -1)
    time = q.shape[0]    
    
    p_maxmin = np.zeros((time,intervals_regular+2,len(latitude),len(longitude)))
    #sp-dp*(0-39) they start with p_maxmin1 = sp
    p_maxmin[:,1:-1,:,:] = sp[:,np.newaxis,:,:] - dp[:,np.newaxis,:,:] * np.arange(0, intervals_regular)[np.newaxis,:,np.newaxis,np.newaxis]
    #values close to the surface will be 1 and higher that P_boundary will be 0
    mask = np.where(p_maxmin > P_boundary[:,np.newaxis,:,:], 1.,0.)
    mask[:,0,:,:] = 1. # bottom value is always 1
    #The 2 lines bellow extract the P_boundary from the numpys and include the values along the column of air 
    p_maxmin[:,:-1,:,:] = mask[:,1:,:,:]*p_maxmin[:,1:,:,:] + (1-mask[:,1:,:,:])*p_maxmin[:,:-1,:,:]
    p_maxmin[:,1:,:,:] = np.where(p_maxmin[:,:-1,:,:] == p_maxmin[:,1:,:,:], P_boundary[:,np.newaxis,:,:], p_maxmin[:,1:,:,:])

    del(dp,mask)
    print ('after p_maxmin and now add surface and atmosphere together for u,q,v', dt.datetime.now().time())    
 
    #In the next steps numpys are created with the information of 32 levels + 2 (surface and 0 Pa)
    levelist = q_levels*100    #Pa   
    p = np.zeros((time, int(levelist.size+2), len(latitude), len(longitude)))
    p[:,1:-1,:,:] = levelist[np.newaxis,:,np.newaxis,np.newaxis]
    p[:,0,:,:] = sp  
 
    u_total = np.zeros((time, levelist.size+2, len(latitude), len(longitude)))
    u_total[:,1:-1,:,:] = u
    u_total[:,0,:,:] = u10
    u_total[:,-1,:,:] = u[:,-1,:,:]
    
    v_total = np.zeros((time, levelist.size+2, len(latitude), len(longitude)))
    v_total[:,1:-1,:,:] = v
    v_total[:,0,:,:] = u10
    v_total[:,-1,:,:] = v[:,-1,:,:]
    
    q_total = np.zeros((time, levelist.size+2, len(latitude), len(longitude)))
    q_total[:,1:-1,:,:] = q
    q_total[:,0,:,:] = q2m
    #mask creates a numpy False are values higher that the surface pressure and True when lev is lower that sp-1000
    mask = np.ones(u_total.shape, dtype=np.bool)
    mask[:,1:-1,:,:] = levelist[np.newaxis,:,np.newaxis,np.newaxis] < (sp[:,np.newaxis,:,:] - 1000.) # Pa
    #these np.ma.masked_array lets the false values to stay and "erase" or "--" to the true values. 
    # the masked numpys will be preserve the lev when they ara lower than the sp-1000 
    u_masked = np.ma.masked_array(u_total, mask=~mask)
    v_masked = np.ma.masked_array(v_total, mask=~mask)
    q_masked = np.ma.masked_array(q_total, mask=~mask)
    p_masked = np.ma.masked_array(p, mask=~mask)
    del(u_total, v_total, q_total, p, u, v, q, u10, q2m, sp)      
       
    print( 'before interpolation loop', dt.datetime.now().time())    
    print('shape',q_masked.shape)
    print(time, intervals_regular+2, len(latitude), len(longitude))
    uq_maxmin = np.zeros((time, intervals_regular+2, len(latitude), len(longitude)))
    vq_maxmin = np.zeros((time, intervals_regular+2, len(latitude), len(longitude)))
    q_maxmin = np.zeros((time, intervals_regular+2, len(latitude), len(longitude)))
    #The loop retrieves the interpolated values along the column of air. It prints the results at pmaxmin levels  
    for t in range(time): #loop over timesteps
        for i in range(len(latitude)): # loop over latitude
            for j in range(len(longitude)): #loop over longitude
                pp = p_masked[t,:,i,j] #take all the values of pressure at each 
                uu = u_masked[t,:,i,j]
                vv = v_masked[t,:,i,j]
                qq = q_masked[t,:,i,j]
                pp = pp[~pp.mask] # filter just the information with values 
                uu = uu[~uu.mask]
                vv = vv[~vv.mask]
                qq = qq[~qq.mask]
                f_uq = interp1d(pp, uu*qq, 'cubic') # spline interpolation
                uq_maxmin[t,:,i,j] = f_uq(p_maxmin[t,:,i,j]) # spline interpolation. The values of uq are filled with the interpolation at the levels of pmaxmin 

                f_vq = interp1d(pp, vv*qq, 'cubic') # spline interpolation
                vq_maxmin[t,:,i,j] = f_vq(p_maxmin[t,:,i,j]) # spline interpolation

                f_q = interp1d(pp, qq) # linear interpolation
                q_maxmin[t,:,i,j] = f_q(p_maxmin[t,:,i,j]) # linear interpolation   
    
    del(u_masked, v_masked, q_masked, p_masked, mask, f_uq, f_vq, f_q)          
    print( 'after interpolation loop', dt.datetime.now().time()    )

    # pressure between full levels. This step makes sure there are no negative values between pressure layers 
    P_between = np.maximum(0, p_maxmin[:,:-1,:,:] - p_maxmin[:,1:,:,:]) # the maximum statement is necessary to avoid negative humidity values
    #Imme: in P_between you do not calculate the pressure between two levels but the pressure difference between two levels!!!
    #Extract the average between two contiguous layers
    q_between = 0.5 * (q_maxmin[:,1:,:,:] + q_maxmin[:,:-1,:,:])
    uq_between = 0.5 * (uq_maxmin[:,1:,:,:] + uq_maxmin[:,:-1,:,:])
    vq_between = 0.5 * (vq_maxmin[:,1:,:,:] + vq_maxmin[:,:-1,:,:])     
    
    #eastward and northward fluxes
    Fa_E_p = uq_between * P_between /g #[kg water/m2]
    Fa_N_p = vq_between * P_between /g 

    # compute the column water vapor 
    cwv = q_between * P_between / g # column water vapor = specific humidity * pressure levels length / g [kg/m2]
    # make tcwv vector
    tcwv = np.squeeze(np.sum(cwv,1)) #total column water vapor, cwv is summed over the vertical [kg/m2]

    #use mask
    mask = np.where(p_maxmin > P_boundary[:,np.newaxis,:,:], 1.,0.)

    vapor_down = np.sum(mask[:,:-1,:,:]*q_between*P_between/g, axis=1)
    vapor_top = np.sum((1-mask[:,:-1,:,:])*q_between*P_between/g, axis=1)

    Fa_E_down = np.sum(mask[:,:-1,:,:]*Fa_E_p, axis=1) #kg*m-1*s-1
    Fa_N_down = np.sum(mask[:,:-1,:,:]*Fa_N_p, axis=1) #kg*m-1*s-1
    Fa_E_top = np.sum((1-mask[:,:-1,:,:])*Fa_E_p, axis=1) #kg*m-1*s-1
    Fa_N_top = np.sum((1-mask[:,:-1,:,:])*Fa_N_p, axis=1) #kg*m-1*s-1

    vapor_total = vapor_top + vapor_down
        
    # check whether the next calculation results in all zeros
    test0 = tcwv - vapor_total
    print('check calculation water vapor, this value should be zero: ' + str(np.sum(test0)))
              
    # put A_gridcell on a 3D grid
    A_gridcell2D = np.tile(A_gridcell,[1,len(longitude)])
    A_gridcell_1_2D = np.reshape(A_gridcell2D, [1,len(latitude),len(longitude)])
    A_gridcell_plus3D = np.tile(A_gridcell_1_2D,[count_time+1,1,1])
    
    # water volumes
    W_top = vapor_top * A_gridcell_plus3D / density_water #[m3]
    W_down = vapor_down * A_gridcell_plus3D / density_water #[m3]  

    return cwv, W_top, W_down, Fa_E_top, Fa_N_top, Fa_E_down, Fa_N_down 

#%% Code 

def getEP(latnrs,lonnrs,yearnumber,begin_time,count_time,latitude,longitude,A_gridcell,model):
    #6-hour data         
    l_vapo = 2450000 #(J/kg)
    #Evaporation is expressed as surface latent heat flux (LHFLX) (J/(s*m^2)) and to transform latent heat of vaporization must
    #be used. then we have in mm/s. To have it in m-- should be multiplied by 3600s * 6h and divided by 1000
    if model=='cesm':
            evaporation = ((Dataset(datapath[6], mode = 'r').variables[var_list[7]][begin_time:(begin_time+count_time),latnrs,lonnrs])/l_vapo)*(3600*6/1000) #m
            precipitation = Dataset(datapath[7], mode = 'r').variables[var_list[8]][begin_time:(begin_time+count_time),latnrs,lonnrs]*6*3600 #m
    else:
            evaporation = ((Dataset(datapath[6], mode = 'r').variables[var_list[7]][begin_time:(begin_time+count_time),latnrs,lonnrs])/l_vapo)*(3600*3/1000) #m
            precipitation = Dataset(datapath[7], mode = 'r').variables[var_list[8]][begin_time:(begin_time+count_time),latnrs,lonnrs]*3*3600/1000 #m
 
    #delete and transfer negative values, change sign convention to all positive
    #For precipitation, they transfer positive values of evaporation to precipitation. 
    #Precipitation will contain the values of precipitation + evaporation (when this is +)
    precipitation = np.reshape(np.maximum(np.reshape(precipitation, (np.size(precipitation))) + np.maximum(np.reshape(evaporation, (np.size(evaporation))),0.0),0.0),
                        (np.int(count_time),len(latitude),len(longitude)))
    #all negative values are changed to positive values 
    evaporation = np.reshape(np.abs(np.minimum(np.reshape(evaporation, (np.size(evaporation))),0.0)),(np.int(count_time),len(latitude),len(longitude)))   
        
    #calculate volumes
    A_gridcell2D = np.tile(A_gridcell,[1,len(longitude)])
    A_gridcell_1_2D = np.reshape(A_gridcell2D, [1,len(latitude),len(longitude)])
    A_gridcell_max3D = np.tile(A_gridcell_1_2D,[count_time,1,1])

    E = evaporation * A_gridcell_max3D #[m3]
    P = precipitation * A_gridcell_max3D #[m3]

    return E, P

#%% Code

# within this new definition of refined I do a linear interpolation over time of my fluxes    
def getrefined_new(Fa_E_top,Fa_N_top,Fa_E_down,Fa_N_down,W_top,W_down,E,P,divt,count_time,latitude,longitude):    
    # This definition refines the timestep of the data
    # Imme: change the timesteps from 6-hourly and 3-hourly to 96 timesteps a day
    # Elena: All the data is in 6-h 
    oddvector = np.zeros((1,np.int(count_time*divt)))
    partvector = np.zeros((1,np.int(count_time*divt)))
    da = np.arange(1,divt) 
    divt = np.float(divt)
    for o in np.arange(0,np.int(count_time*divt),np.int(divt)):
        for i in range(len(da)):
            oddvector[0,o+i]    = (divt-da[i])/divt
            partvector[0,o+i+1] = da[i]/divt    
        
    W_top_small = np.nan*np.zeros((np.int(count_time*divt+1),len(latitude),len(longitude)))
    W_down_small = np.nan*np.zeros((np.int(count_time*divt+1),len(latitude),len(longitude)))        

    Fa_E_down_small = np.nan*np.zeros((np.int(count_time*divt),len(latitude),len(longitude)))
    Fa_N_down_small = np.nan*np.zeros((np.int(count_time*divt),len(latitude),len(longitude)))
    Fa_E_top_small = np.nan*np.zeros((np.int(count_time*divt),len(latitude),len(longitude)))
    Fa_N_top_small = np.nan*np.zeros((np.int(count_time*divt),len(latitude),len(longitude)))
    E_small = np.nan*np.zeros((np.int(count_time*divt),len(latitude),len(longitude)))
    P_small = np.nan*np.zeros((np.int(count_time*divt),len(latitude),len(longitude)))

    for t in range(1,np.int((count_time)*divt)+1):
        W_top_small[t-1] = W_top[np.int(t/divt+oddvector[0,t-1]-1)] + partvector[0,t-1] * (W_top[np.int(t/divt+oddvector[0,t-1])] - W_top[np.int(t/divt+oddvector[0,t-1]-1)])
        W_top_small[-1] = W_top[-1]
        W_down_small[t-1] = W_down[np.int(t/divt+oddvector[0,t-1]-1)] + partvector[0,t-1] * (W_down[np.int(t/divt+oddvector[0,t-1])] - W_down[np.int(t/divt+oddvector[0,t-1]-1)])
        W_down_small[-1] = W_down[-1]

        Fa_E_down_small[t-1] = Fa_E_down[np.int(t/divt+oddvector[0,t-1]-1)] + partvector[0,t-1] * (Fa_E_down[np.int(t/divt+oddvector[0,t-1])] - Fa_E_down[np.int(t/divt+oddvector[0,t-1]-1)])
        Fa_N_down_small[t-1] = Fa_N_down[np.int(t/divt+oddvector[0,t-1]-1)] + partvector[0,t-1] * (Fa_N_down[np.int(t/divt+oddvector[0,t-1])] - Fa_N_down[np.int(t/divt+oddvector[0,t-1]-1)])
        Fa_E_top_small[t-1] = Fa_E_top[np.int(t/divt+oddvector[0,t-1]-1)] + partvector[0,t-1] * (Fa_E_top[np.int(t/divt+oddvector[0,t-1])] - Fa_E_top[np.int(t/divt+oddvector[0,t-1]-1)])
        Fa_N_top_small[t-1] = Fa_N_top[np.int(t/divt+oddvector[0,t-1]-1)] + partvector[0,t-1] * (Fa_N_top[np.int(t/divt+oddvector[0,t-1])] - Fa_N_top[np.int(t/divt+oddvector[0,t-1]-1)])
        P_small[t-1] = (1./divt) * P[np.int(t/divt+oddvector[0,t-1]-1)]
        E_small[t-1] = (1./divt) * E[np.int(t/divt+oddvector[0,t-1]-1)]
        
    W_top = W_top_small
    W_down = W_down_small
    Fa_E_down = Fa_E_down_small
    Fa_N_down = Fa_N_down_small
    Fa_E_top = Fa_E_top_small
    Fa_N_top = Fa_N_top_small
    P = P_small
    E = E_small 
    
    return Fa_E_top,Fa_N_top,Fa_E_down,Fa_N_down,E,P,W_top,W_down

#%% Code
def change_units(Fa_E_top_1,Fa_E_down_1,Fa_N_top_1,Fa_N_down_1,timestep,divt,L_EW_gridcell,density_water,L_N_gridcell,L_S_gridcell,latitude):
    
    #redefine according to units
    Fa_E_top_kgpmps = Fa_E_top_1
    Fa_E_down_kgpmps = Fa_E_down_1
    Fa_N_top_kgpmps = Fa_N_top_1
    Fa_N_down_kgpmps = Fa_N_down_1
    
    #convert to m3
    Fa_E_top_m3 = Fa_E_top_kgpmps * timestep/np.float(divt) * L_EW_gridcell / density_water # [kg*m^-1*s^-1*s*m*kg^-1*m^3]=[m3]
    Fa_E_down_m3 = Fa_E_down_kgpmps * timestep/np.float(divt) * L_EW_gridcell / density_water # [s*m*kg*m^-1*s^-1*kg^-1*m^3]=[m3]
      
    Fa_N_top_swap = np.zeros((len(latitude),np.int(count_time*np.float(divt)),len(longitude)))
    Fa_N_down_swap = np.zeros((len(latitude),np.int(count_time*np.float(divt)),len(longitude)))
    Fa_N_top_kgpmps_swap = np.swapaxes(Fa_N_top_kgpmps,0,1)
    Fa_N_down_kgpmps_swap = np.swapaxes(Fa_N_down_kgpmps,0,1)
    for c in range(len(latitude)):
        Fa_N_top_swap[c] = Fa_N_top_kgpmps_swap[c] * timestep/np.float(divt) * 0.5 *(L_N_gridcell[c]+L_S_gridcell[c]) / density_water # [s*m*kg*m^-1*s^-1*kg^-1*m^3]=[m3]
        Fa_N_down_swap[c] = Fa_N_down_kgpmps_swap[c] * timestep/np.float(divt) * 0.5*(L_N_gridcell[c]+L_S_gridcell[c]) / density_water # [s*m*kg*m^-1*s^-1*kg^-1*m^3]=[m3]

    Fa_N_top_m3 = np.swapaxes(Fa_N_top_swap,0,1) 
    Fa_N_down_m3 = np.swapaxes(Fa_N_down_swap,0,1) 

    return Fa_E_top_m3, Fa_E_down_m3, Fa_N_top_m3, Fa_N_down_m3                                   

def get_stablefluxes(Fa_E_top,Fa_E_down,Fa_N_top,Fa_N_down,
                                   timestep,divt,L_EW_gridcell,density_water,L_N_gridcell,L_S_gridcell,latitude):
    
    #find out where the negative fluxes are
    Fa_E_top_posneg = np.ones(np.shape(Fa_E_top))
    Fa_E_top_posneg[Fa_E_top < 0] = -1
    Fa_N_top_posneg = np.ones(np.shape(Fa_E_top))
    Fa_N_top_posneg[Fa_N_top < 0] = -1
    Fa_E_down_posneg = np.ones(np.shape(Fa_E_top))
    Fa_E_down_posneg[Fa_E_down < 0] = -1
    Fa_N_down_posneg = np.ones(np.shape(Fa_E_top))
    Fa_N_down_posneg[Fa_N_down < 0] = -1
    
    #make everything absolute   
    Fa_E_top_abs = np.abs(Fa_E_top)
    Fa_E_down_abs = np.abs(Fa_E_down)
    Fa_N_top_abs = np.abs(Fa_N_top)
    Fa_N_down_abs = np.abs(Fa_N_down)
    
    # stabilize the outfluxes / influxes
    stab = 1./2.  # during the reduced timestep the water cannot move further than 1/x * the gridcell, 
                    #in other words at least x * the reduced timestep is needed to cross a gridcell
    Fa_E_top_stable = np.reshape(np.minimum(np.reshape(Fa_E_top_abs, (np.size(Fa_E_top_abs))), (np.reshape(Fa_E_top_abs, (np.size(Fa_E_top_abs)))  / 
                                    (np.reshape(Fa_E_top_abs, (np.size(Fa_E_top_abs)))  + np.reshape(Fa_N_top_abs, (np.size(Fa_N_top_abs))))) * stab 
                                            * np.reshape(W_top[:-1,:,:], (np.size(W_top[:-1,:,:])))),(np.int(count_time*np.float(divt)),len(latitude),len(longitude)))
    Fa_N_top_stable = np.reshape(np.minimum(np.reshape(Fa_N_top_abs, (np.size(Fa_N_top_abs))), (np.reshape(Fa_N_top_abs, (np.size(Fa_N_top_abs)))  / 
                                    (np.reshape(Fa_E_top_abs, (np.size(Fa_E_top_abs)))  + np.reshape(Fa_N_top_abs, (np.size(Fa_N_top_abs))))) * stab 
                                            * np.reshape(W_top[:-1,:,:], (np.size(W_top[:-1,:,:])))),(np.int(count_time*np.float(divt)),len(latitude),len(longitude)))
    Fa_E_down_stable = np.reshape(np.minimum(np.reshape(Fa_E_down_abs, (np.size(Fa_E_down_abs))), (np.reshape(Fa_E_down_abs, (np.size(Fa_E_down_abs)))  / 
                                    (np.reshape(Fa_E_down_abs, (np.size(Fa_E_down_abs)))  + np.reshape(Fa_N_down_abs, (np.size(Fa_N_down_abs))))) * stab 
                                             * np.reshape(W_down[:-1,:,:], (np.size(W_down[:-1,:,:])))),(np.int(count_time*np.float(divt)),len(latitude),len(longitude)))
    Fa_N_down_stable = np.reshape(np.minimum(np.reshape(Fa_N_down_abs, (np.size(Fa_N_down_abs))), (np.reshape(Fa_N_down_abs, (np.size(Fa_N_down_abs)))  / 
                                    (np.reshape(Fa_E_down_abs, (np.size(Fa_E_down_abs)))  + np.reshape(Fa_N_down_abs, (np.size(Fa_N_down_abs))))) * stab 
                                             * np.reshape(W_down[:-1,:,:], (np.size(W_down[:-1,:,:])))),(np.int(count_time*np.float(divt)),len(latitude),len(longitude)))
    
    #get rid of the nan values
    Fa_E_top_stable[np.isnan(Fa_E_top_stable)] = 0
    Fa_N_top_stable[np.isnan(Fa_N_top_stable)] = 0
    Fa_E_down_stable[np.isnan(Fa_E_down_stable)] = 0
    Fa_N_down_stable[np.isnan(Fa_N_down_stable)] = 0
    
    #redefine
    Fa_E_top = Fa_E_top_stable * Fa_E_top_posneg
    Fa_N_top = Fa_N_top_stable * Fa_N_top_posneg
    Fa_E_down = Fa_E_down_stable * Fa_E_down_posneg
    Fa_N_down = Fa_N_down_stable * Fa_N_down_posneg
    
    return Fa_E_top, Fa_E_down, Fa_N_top, Fa_N_down

#%% Code
def getFa_Vert(Fa_E_top,Fa_E_down,Fa_N_top,Fa_N_down,E,P,W_top,W_down,divt,count_time,latitude,longitude):
    
    #total moisture in the column
    W = W_top + W_down
    
    #define the horizontal fluxes over the boundaries
    # fluxes over the eastern boundary
    Fa_E_top_boundary = np.zeros(np.shape(Fa_E_top))
    Fa_E_top_boundary[:,:,:-1] = 0.5 * (Fa_E_top[:,:,:-1] + Fa_E_top[:,:,1:])
    Fa_E_down_boundary = np.zeros(np.shape(Fa_E_down))
    Fa_E_down_boundary[:,:,:-1] = 0.5 * (Fa_E_down[:,:,:-1] + Fa_E_down[:,:,1:])

    # find out where the positive and negative fluxes are
    Fa_E_top_pos = np.ones(np.shape(Fa_E_top))
    Fa_E_down_pos = np.ones(np.shape(Fa_E_down))
    Fa_E_top_pos[Fa_E_top_boundary < 0] = 0
    Fa_E_down_pos[Fa_E_down_boundary < 0] = 0
    Fa_E_top_neg = Fa_E_top_pos - 1
    Fa_E_down_neg = Fa_E_down_pos - 1

    # separate directions west-east (all positive numbers)
    Fa_E_top_WE = Fa_E_top_boundary * Fa_E_top_pos;
    Fa_E_top_EW = Fa_E_top_boundary * Fa_E_top_neg;
    Fa_E_down_WE = Fa_E_down_boundary * Fa_E_down_pos;
    Fa_E_down_EW = Fa_E_down_boundary * Fa_E_down_neg;

    # fluxes over the western boundary
    Fa_W_top_WE = np.nan*np.zeros(np.shape(P))
    Fa_W_top_WE[:,:,1:] = Fa_E_top_WE[:,:,:-1]
    Fa_W_top_WE[:,:,0] = Fa_E_top_WE[:,:,-1]
    Fa_W_top_EW = np.nan*np.zeros(np.shape(P))
    Fa_W_top_EW[:,:,1:] = Fa_E_top_EW[:,:,:-1]
    Fa_W_top_EW[:,:,0] = Fa_E_top_EW[:,:,-1]
    Fa_W_down_WE = np.nan*np.zeros(np.shape(P))
    Fa_W_down_WE[:,:,1:] = Fa_E_down_WE[:,:,:-1]
    Fa_W_down_WE[:,:,0] = Fa_E_down_WE[:,:,-1]
    Fa_W_down_EW = np.nan*np.zeros(np.shape(P))
    Fa_W_down_EW[:,:,1:] = Fa_E_down_EW[:,:,:-1]
    Fa_W_down_EW[:,:,0] = Fa_E_down_EW[:,:,-1]    

    # fluxes over the northern boundary
    Fa_N_top_boundary = np.nan*np.zeros(np.shape(Fa_N_top));
    Fa_N_top_boundary[:,1:,:] = 0.5 * ( Fa_N_top[:,:-1,:] + Fa_N_top[:,1:,:] )
    Fa_N_down_boundary = np.nan*np.zeros(np.shape(Fa_N_down));
    Fa_N_down_boundary[:,1:,:] = 0.5 * ( Fa_N_down[:,:-1,:] + Fa_N_down[:,1:,:] )

    # find out where the positive and negative fluxes are
    Fa_N_top_pos = np.ones(np.shape(Fa_N_top))
    Fa_N_down_pos = np.ones(np.shape(Fa_N_down))
    Fa_N_top_pos[Fa_N_top_boundary < 0] = 0
    Fa_N_down_pos[Fa_N_down_boundary < 0] = 0
    Fa_N_top_neg = Fa_N_top_pos - 1
    Fa_N_down_neg = Fa_N_down_pos - 1

    # separate directions south-north (all positive numbers)
    Fa_N_top_SN = Fa_N_top_boundary * Fa_N_top_pos
    Fa_N_top_NS = Fa_N_top_boundary * Fa_N_top_neg
    Fa_N_down_SN = Fa_N_down_boundary * Fa_N_down_pos
    Fa_N_down_NS = Fa_N_down_boundary * Fa_N_down_neg

    # fluxes over the southern boundary
    Fa_S_top_SN = np.nan*np.zeros(np.shape(P))
    Fa_S_top_SN[:,:-1,:] = Fa_N_top_SN[:,1:,:]
    Fa_S_top_NS = np.nan*np.zeros(np.shape(P))
    Fa_S_top_NS[:,:-1,:] = Fa_N_top_NS[:,1:,:]
    Fa_S_down_SN = np.nan*np.zeros(np.shape(P))
    Fa_S_down_SN[:,:-1,:] = Fa_N_down_SN[:,1:,:]
    Fa_S_down_NS = np.nan*np.zeros(np.shape(P))
    Fa_S_down_NS[:,:-1,:] = Fa_N_down_NS[:,1:,:]

    # check the water balance
    Sa_after_Fa_down = np.zeros([1,len(latitude),len(longitude)])
    Sa_after_Fa_top = np.zeros([1,len(latitude),len(longitude)])
    Sa_after_all_down = np.zeros([1,len(latitude),len(longitude)])
    Sa_after_all_top = np.zeros([1,len(latitude),len(longitude)])
    residual_down = np.zeros(np.shape(P)) # residual factor [m3]
    residual_top = np.zeros(np.shape(P)) # residual factor [m3]

    for t in range(np.int(count_time*divt)):
        # down: calculate with moisture fluxes:
        Sa_after_Fa_down[0,1:-1,:] = (W_down[t,1:-1,:] - Fa_E_down_WE[t,1:-1,:] + Fa_E_down_EW[t,1:-1,:] + Fa_W_down_WE[t,1:-1,:] - Fa_W_down_EW[t,1:-1,:] - Fa_N_down_SN[t,1:-1,:] + Fa_N_down_NS[t,1:-1,:] + Fa_S_down_SN[t,1:-1,:] - Fa_S_down_NS[t,1:-1,:])

        # top: calculate with moisture fluxes:
        Sa_after_Fa_top[0,1:-1,:] = (W_top[t,1:-1,:]- Fa_E_top_WE[t,1:-1,:] + Fa_E_top_EW[t,1:-1,:] + Fa_W_top_WE[t,1:-1,:] - Fa_W_top_EW[t,1:-1,:] - Fa_N_top_SN[t,1:-1,:] + Fa_N_top_NS[t,1:-1,:] + Fa_S_top_SN[t,1:-1,:]- Fa_S_top_NS[t,1:-1,:])
    
        # down: substract precipitation and add evaporation
        Sa_after_all_down[0,1:-1,:] = Sa_after_Fa_down[0,1:-1,:] - P[t,1:-1,:] * (W_down[t,1:-1,:] / W[t,1:-1,:]) + E[t,1:-1,:]
    
        # top: substract precipitation
        Sa_after_all_top[0,1:-1,:] = Sa_after_Fa_top[0,1:-1,:] - P[t,1:-1,:] * (W_top[t,1:-1,:] / W[t,1:-1,:])
    
        # down: calculate the residual
        residual_down[t,1:-1,:] = W_down[t+1,1:-1,:] - Sa_after_all_down[0,1:-1,:]
    
        # top: calculate the residual
        residual_top[t,1:-1,:] = W_top[t+1,1:-1,:] - Sa_after_all_top[0,1:-1,:]

    # compute the resulting vertical moisture flux
    Fa_Vert_raw = W_down[1:,:,:] / W[1:,:,:] * (residual_down + residual_top) - residual_down # the vertical velocity so that the new residual_down/W_down =  residual_top/W_top (positive downward)

    # find out where the negative vertical flux is
    Fa_Vert_posneg = np.ones(np.shape(Fa_Vert_raw))
    Fa_Vert_posneg[Fa_Vert_raw < 0] = -1

    # make the vertical flux absolute
    Fa_Vert_abs = np.abs(Fa_Vert_raw)

    # stabilize the outfluxes / influxes
    stab = 1./4. #during the reduced timestep the vertical flux can maximally empty/fill 1/x of the top or down storage
    
    Fa_Vert_stable = np.reshape(np.minimum(np.reshape(Fa_Vert_abs, (np.size(Fa_Vert_abs))), np.minimum(stab*np.reshape(W_top[1:,:,:], (np.size(W_top[1:,:,:]))), stab*np.reshape(W_down[1:,:,:], (np.size(W_down[1:,:,:]))))),(np.int(count_time*np.float(divt)),len(latitude),len(longitude)))
                
    # redefine the vertical flux
    Fa_Vert = Fa_Vert_stable * Fa_Vert_posneg;

    return Fa_Vert_raw, Fa_Vert, residual_down, residual_top

#%% Runtime & Results
os.chdir(r'/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/wam2layer/scripts')
#start1 = timer()

# obtain the constants
latitude,longitude,lsm,g,density_water,timestep,A_gridcell,L_N_gridcell,L_S_gridcell,L_EW_gridcell,gridcell = \
    getconstants_pressure_CESM(model,latnrs,lonnrs,lsm_data_CESM,area_mask)

a=1
yearnumber = start_year
monthnumber = 1
            
datapath = data_path(yearnumber,monthnumber,a,model)
if model =='cesm':
    if start_month!=1:
        coef = (yearnumber-130)*count_time*365+np.sum(months_length_nonleap[:start_month-1])*count_time #This coefficient helps when the first day is not the index of the nedcdf 
    else:
        coef = (yearnumber-130)*count_time*365
else:
    if start_month!=1:
        if calendar.isleap(start_year):
            coef = (yearnumber-2145)*count_time*365+np.sum(months_length_leap[:start_month-1])*count_time #This coefficient helps when the first day is 
        else:
            coef = (yearnumber-2145)*count_time*365+np.sum(months_length_nonleap[:start_month-1])*count_time #This coefficient helps when the first day is not t$
    else:
        coef = (yearnumber-2145)*count_time*365

old_month=start_month
i=0
for date in datelist[:]:
   # start = timer()        
    a=date.day
    yearnumber = date.year
    monthnumber = date.month
    
    print(monthnumber)
    if i>0:
        old_month=datelist[datelist.index(date)-1].month
    i=i+1
    print(old_month)
    if old_month !=monthnumber:
       if model =='cesm':
           coef=coef+months_length_nonleap[old_month-1]
       else:
           if calendar.isleap(yearnumber):
               coef=coef+months_length_leap[old_month-1]
           else:
               coef=coef+months_length_nonleap[old_month-1]

    begin_time = (a-1)*count_time + coef # because python starts counting at 0 (so the first timesteps start at 0)
    final_time = months_length_nonleap[monthnumber-1]

   
    print( date, yearnumber, monthnumber, a, begin_time, final_time    )
    print(latnrs,lonnrs,a,yearnumber,begin_time,count_time,density_water)
    #print(Dataset(datapath[0], mode = 'r').variables[var_list[0]][begin_time:(begin_time+count_time+1),::-1,latnrs,lonnrs])
    print(Dataset(datapath[0], mode = 'r').variables[var_list[0]])
    print(begin_time,begin_time+count_time+1,latnrs,lonnrs)
    #print('0 = ' + str(timer()))
#    #1 integrate specific humidity to get the (total) column water (vapor) and calculate horizontal moisture fluxes
    cwv, W_top, W_down, Fa_E_top, Fa_N_top, Fa_E_down, Fa_N_down = \
        getWandFluxes(latnrs,lonnrs,a,yearnumber,begin_time,count_time,density_water,latitude,longitude,g,A_gridcell,model)
    #print('1,2,3 = ' + str(timer()))    
                
    #4 evaporation and precipitation
    E,P = getEP(latnrs,lonnrs,yearnumber,begin_time,count_time,latitude,longitude,A_gridcell)
    #print('4 = ' + str(timer()))
                
    # put data on a smaller time step
    Fa_E_top_1,Fa_N_top_1,Fa_E_down_1,Fa_N_down_1,E,P,W_top,W_down = getrefined_new(Fa_E_top,Fa_N_top,Fa_E_down,Fa_N_down,W_top,W_down,E,P,divt,count_time,latitude,longitude)
    #print('5 = ' + str(timer()))    
   
    # change units to m3
    Fa_E_top_m3,Fa_E_down_m3,Fa_N_top_m3,Fa_N_down_m3 = change_units(Fa_E_top_1,Fa_E_down_1,Fa_N_top_1,Fa_N_down_1,
                               timestep,divt,L_EW_gridcell,density_water,L_N_gridcell,L_S_gridcell,latitude)
    #print('6a = ' + str(timer()))
    
    #Before horizontal fluxes I saved the data to avoid Memory problems
    #np.savez(datapath[1], Fa_E_top_m3=Fa_E_top_m3, Fa_N_top_m3=Fa_N_top_m3, Fa_E_down_m3=Fa_E_down_m3,Fa_N_down_m3=Fa_N_down_m3, E=E, P=P, W_top=W_top, W_down=W_down) # save as np file
    #data = np.load(datapath[1]+'.npz')
    #Fa_E_top_m3, Fa_N_top_m3, Fa_E_down_m3,Fa_N_down_m3, E, P,W_top, W_down = data['Fa_E_top_m3'], data['Fa_N_top_m3'], data['Fa_E_down_m3'],data['Fa_N_down_m3'],data['E'],data['P'],data['W_top'],data['W_down']
         
    # stabilize horizontal fluxes
    Fa_E_top,Fa_E_down,Fa_N_top,Fa_N_down = get_stablefluxes(Fa_E_top_m3,Fa_E_down_m3,Fa_N_top_m3,Fa_N_down_m3,
                               timestep,divt,L_EW_gridcell,density_water,L_N_gridcell,L_S_gridcell,latitude)
    #print('6b = ' + str(timer()))
                               
    # determine the vertical moisture flux
    Fa_Vert_raw,Fa_Vert, residual_down, residual_top = getFa_Vert(Fa_E_top,Fa_E_down,Fa_N_top,Fa_N_down,E,P,W_top,W_down,divt,count_time,latitude,longitude)
    #print('7 = ' + str(timer()))

    datapath1 = data_path(yearnumber,monthnumber,a,model)
    #np.savez_compressed(datapath[16], E=E, P=P, Fa_E_top=Fa_E_top, Fa_N_top= Fa_N_top, Fa_E_down=Fa_E_down, Fa_N_down=Fa_N_down, W_down=W_down, W_top=W_top, residual_top=residual_top, residual_down=residual_down, Fa_Vert=Fa_Vert) # save as .npy file   
    
    sio.savemat(datapath1[9], {'Fa_E_top':Fa_E_top, 'Fa_N_top':Fa_N_top, 'Fa_E_down':Fa_E_down,'Fa_N_down':Fa_N_down, 'E':E, 'P':P, 
                                                                                    'W_top':W_top, 'W_down':W_down, 'Fa_Vert':Fa_Vert}, do_compression=True) # save as mat file    

    #end = timer()
   # print ('Runtime fluxes_and_storages for day ' + str(a) + ' in year ' + str(yearnumber) + ' is',(end - start),' seconds.')
#end1 = timer()
#print ('The total runtime is',(end1-start1),' seconds.')
print('Procces finished')


