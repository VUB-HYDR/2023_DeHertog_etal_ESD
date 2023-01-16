from __future__ import print_function
import sys
import os
from getpass import getuser
import string
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import netCDF4 as netcdf4
import xarray as xr
import pandas
#import regionmask
import cartopy.crs as ccrs
#from IPython.display import display, Math, Latex
import warnings
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import transforms
#from mask import mask_data

outdir = '/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/cesm/work/postprocessing/signals_seperated/'
procdir=outdir
#os.chdir(procdir)

case_irr_ctl   ='irr-ctl'
case_irr_crop ='irr-crop'
case_crop_ctl ='crop-ctl'
case_frst_ctl ='frst-ctl'

def plot_mon_hovmoller(var,case,outdir,vlims=False,cmor_table='Amon',model='cesm',mask=False):
    varname=var
    var_local=var+'_local'
    var_nonlocal=var+'_nonlocal'
    var_total=var+'_total'
    
    tseriesdir= outdir + '/'+case+'/' +cmor_table+'/' + varname +'/' 
    fn1 = varname + '_' + case +'_' + model +'_ymonmean000000_signal-separated.nc'
    fn2 = varname + '_' + case +'_' + model +'_ymonmean000001_signal-separated.nc'
    fn3 = varname + '_' + case +'_' + model +'_ymonmean000002_signal-separated.nc'
    fn4 = varname + '_' + case +'_' + model +'_ymonmean000003_signal-separated.nc'
    fn5 = varname + '_' + case +'_' + model +'_ymonmean000004_signal-separated.nc'
    
    f, ax = plt.subplots(3,1,sharex=True)
    ax = ax.flatten()
    k=0
    ens=[0,1,2,3,4,5]
    if mask==False and model =='cesm':
        ensmean_local=np.zeros((12,192))
        ensmean_nonlocal=np.zeros((12,192))
        ensmean_total=np.zeros((12,192))
    elif mask!=False and model=='cesm':
        ensmean_local=np.zeros((192,12))
        ensmean_nonlocal=np.zeros((192,12))
        ensmean_total=np.zeros((192,12))
    elif mask!=False and model=='mpiesm':
        ensmean_local=np.zeros((96,12))
        ensmean_nonlocal=np.zeros((96,12))
        ensmean_total=np.zeros((96,12))
    elif mask==False and model=='mpiesm':
        ensmean_local=np.zeros((12,96))
        ensmean_nonlocal=np.zeros((12,96))
        ensmean_total=np.zeros((12,96))
    for fn in [fn1,fn2,fn3,fn4,fn5]:
        k=k+1
        
        # open the dataset
        ds = xr.open_dataset(tseriesdir+fn)
        #ds_mask = xr.open_dataset(mask)
        da_local=ds[var_local]
        da_nonlocal=ds[var_nonlocal]
        da_total=ds[var_total]
        name=da_total.name
        #da_mask=ds_mask['landmask']
        i=0
        for da in [da_local, da_nonlocal, da_total]:
           i=i+1
           da=mask_data(da,case,model,mask)

           # annual means already taken
           da=da.mean(dim=('lon'))
        # plot data array as a map, with the previously defined argumnets
           if i==1:
               ensmean_local=ensmean_local+da.values
           elif i==2:
              ensmean_nonlocal=ensmean_nonlocal+da.values
           elif i==3:
              ensmean_total=ensmean_total+da.values
          # set the extent of the cartopy geoAxes to \"global\"
          #ax.set_global()
    
          # or alternatively, if you want to plot a certain region, use (example: Europe)
          #ax.set_extent([-180, 180, -63, 89], ccrs.PlateCarree())
          # ax.set_extent([-13, 43, 35, 70], ccrs.PlateCarree())
          # save the figure, adjusting the resolution 
    if mask==False and vlims!=False:
        da['time']=np.arange(1,13,1)
        levels = np.linspace(vlims[0],vlims[1],11)

        h=ax[0].contourf(da.lat,da.time,ensmean_local/5, vmin = vlims[0], vmax = vlims[1],cmap='coolwarm',levels=levels,extend='both')
                    # add the title
        ax[0].set_aspect('auto')
        h=ax[1].contourf(da.lat,da.time,ensmean_nonlocal/5, vmin = vlims[0], vmax = vlims[1],cmap='coolwarm',levels=levels,extend='both')
                    # add the title
        ax[1].set_aspect('auto')
        h=ax[2].contourf(da.lat,da.time,ensmean_total/5, vmin = vlims[0], vmax = vlims[1],cmap='coolwarm',levels=levels,extend='both')
                    # add the title
        ax[2].set_aspect('auto')
    elif mask!=False and vlims!=False:
        levels = np.linspace(vlims[0],vlims[1],11)
        da['time']=np.arange(1,13,1)
        h=ax[0].contourf(da.time,da.lat,ensmean_local/5, vmin = vlims[0], vmax = vlims[1],levels=levels,extend='both')
                    # add the title
        ax[0].set_aspect('auto')
        h=ax[1].contourf(da.time,da.lat,ensmean_nonlocal/5, vmin = vlims[0], vmax = vlims[1],levels=levels,extend='both')
                    # add the title
        ax[1].set_aspect('auto')
        h=ax[2].contourf(da.time,da.lat,ensmean_total/5, vmin = vlims[0], vmax = vlims[1],levels=levels,extend='both')
                    # add the title
        ax[2].set_aspect('auto')
    if mask==False and vlims==False:
        da['time']=np.arange(1,13,1)
        h=ax[0].contourf(da.lat,da.time,ensmean_local/5)
                    # add the title
        ax[0].set_aspect('auto')
        h=ax[1].contourf(da.lat,da.time,ensmean_nonlocal/5)
                    # add the title
        ax[1].set_aspect('auto')
        h=ax[2].contourf(da.lat,da.time,ensmean_total/5)
                    # add the title
        ax[2].set_aspect('auto')
    elif mask!=False and vlims==False:
        da['time']=np.arange(1,13,1)
        h=ax[0].contourf(da.time,da.lat,ensmean_local/5)
                    # add the title
        ax[0].set_aspect('auto')
        h=ax[1].contourf(da.time,da.lat,ensmean_nonlocal/5)
                    # add the title
        ax[1].set_aspect('auto')
        h=ax[2].contourf(da.time,da.lat,ensmean_total/5)
                    # add the title
        ax[2].set_aspect('auto')
    f.colorbar(h, ax=ax)

    plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' + name+'_'+'hovmoller_'+model+'.png', dpi=300)
         
    plt.show()



def plot_diurnal(var,case,outdir,cmor_table='3hr',model='cesm',mask=False):
    varname=var
    var_local=var+'_local'
    var_nonlocal=var+'_nonlocal'
    var_total=var+'_total'
    
    tseriesdir= outdir + '/'+case+'/' +cmor_table+'/' + varname +'/' 
   # fn = varname + '_' + case +'_' + model + '_signal-separated_absolute_values_'+case.split('-')[0]+ '-climate_merge.nc'
    fn1 = varname + '_' + case +'_' + model +'_yhourmean000000_signal-separated.nc'
    fn2 = varname + '_' + case +'_' + model +'_yhourmean000001_signal-separated.nc'
    fn3 = varname + '_' + case +'_' + model +'_yhourmean000002_signal-separated.nc'
    fn4 = varname + '_' + case +'_' + model +'_yhourmean000003_signal-separated.nc'
    fn5 = varname + '_' + case +'_' + model +'_yhourmean000004_signal-separated.nc'

    f, ax = plt.subplots(3,1,sharex=True)
    ax = ax.flatten()
    k=0
    ens=[0,1,2,3,4,5]
    for fn in [fn1,fn2,fn3,fn4,fn5]:
        k=k+1
        # open the dataset
        ds = xr.open_dataset(tseriesdir+fn)
        #ds_mask = xr.open_dataset(mask)
        da_local=ds[var_local]
        da_nonlocal=ds[var_nonlocal]
        da_total=ds[var_total]
        #da_mask=ds_mask['landmask']
        i=0
        for da in [da_local, da_nonlocal, da_total]:
           i=i+1
           name=da.name
           da=mask_data(da,case,model,mask)

           # annual means already taken
           da=da.mean(dim=['lat','lon']).groupby("time.hour").mean()
           #da['time']=np.arange(1,10,1)
           # title 
           title = da.name
           
           # initiate the figure 
           xlims = [0,24]
           values_shifted=np.zeros(12)
           values=da.values
           values_shifted[0:5]=values[7:]
           values_shifted[5:]=values[0:7]
           #time=da.time
           
        # plot data array as a map, with the previously defined argumnets
           if i==1:
              h=da.plot(ax=ax[0],xlim=xlims,label=str(ens[k]))
              #ax[0].plot(time,values_shifted,label=str(ens[k]))
                # add the title
              ax[0].set_title(title)
              ax[0].set_aspect('auto')
           elif i==2:
              h=da.plot(ax=ax[1],xlim=xlims,label=str(ens[k]))
              #ax[1].plot(time,values_shifted,label=str(ens[k]))
                # add the title
              ax[1].set_title(title)
              ax[1].set_aspect('auto')
           elif i==3:
              h=da.plot(ax=ax[2],xlim=xlims,label=str(ens[k]))
              #ax[2].plot(time,values_shifted,label=str(ens[k]))
                # add the title
              ax[2].set_title(title)
              ax[2].set_aspect('auto')

          # set the extent of the cartopy geoAxes to \"global\"
          #ax.set_global()
    
          # or alternatively, if you want to plot a certain region, use (example: Europe)
          #ax.set_extent([-180, 180, -63, 89], ccrs.PlateCarree())
          # ax.set_extent([-13, 43, 35, 70], ccrs.PlateCarree())
          # save the figure, adjusting the resolution 
    plt.legend()
    f.tight_layout()

    plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' + name+'_'+'diurnalcycle_'+model+'.png', dpi=300)
         
    plt.show()


def plot_mon_seascycle(var,case,outdir,cmor_table='Amon',model='cesm',mask=False):
    varname=var
    var_local=var+'_local'
    var_nonlocal=var+'_nonlocal'
    var_total=var+'_total'
    
    tseriesdir= outdir + '/'+case+'/' +cmor_table+'/' + varname +'/' 
   # fn = varname + '_' + case +'_' + model + '_signal-separated_absolute_values_'+case.split('-')[0]+ '-climate_merge.nc'
    fn1 = varname + '_' + case +'_' + model +'_ymonmean000000_signal-separated.nc'
    fn2 = varname + '_' + case +'_' + model +'_ymonmean000001_signal-separated.nc'
    fn3 = varname + '_' + case +'_' + model +'_ymonmean000002_signal-separated.nc'
    fn4 = varname + '_' + case +'_' + model +'_ymonmean000003_signal-separated.nc'
    fn5 = varname + '_' + case +'_' + model +'_ymonmean000004_signal-separated.nc'

    f, ax = plt.subplots(3,1,sharex=True)
    ax = ax.flatten()
    k=0
    ens=[0,1,2,3,4,5]
    for fn in [fn1,fn2,fn3,fn4,fn5]:
        k=k+1
        # open the dataset
        ds = xr.open_dataset(tseriesdir+fn)
        #ds_mask = xr.open_dataset(mask)
        da_local=ds[var_local]
        da_nonlocal=ds[var_nonlocal]
        da_total=ds[var_total]
        #da_mask=ds_mask['landmask']
        i=0
        for da in [da_local, da_nonlocal, da_total]:
           i=i+1
           da=mask_data(da,case,model,mask)
           # annual means already taken
           da=da.mean(dim=('lon','lat'))
           da['time']=np.arange(1,13,1)
           # title 
           title = da.name
           name=da.name
           # initiate the figure 
           xlims = [0,13]
           values_shifted=np.zeros(12)
           values=da.values
           values_shifted[0:5]=values[7:]
           values_shifted[5:]=values[0:7]
           time=da.time
           
        # plot data array as a map, with the previously defined argumnets
           if i==1:
              #h=da.plot(ax=ax[0],xlim=xlims,label=str(ens[k]))
              ax[0].plot(time,values_shifted,label=str(ens[k]))
                # add the title
              ax[0].set_title(title)
              ax[0].set_aspect('auto')
           elif i==2:
              #h=da.plot(ax=ax[0],xlim=xlims,label=str(ens[k]))
              ax[1].plot(time,values_shifted,label=str(ens[k]))
                # add the title
              ax[1].set_title(title)
              ax[1].set_aspect('auto')
           elif i==3:
              #h=da.plot(ax=ax[0],xlim=xlims,label=str(ens[k]))
              ax[2].plot(time,values_shifted,label=str(ens[k]))
                # add the title
              ax[2].set_title(title)
              ax[2].set_aspect('auto')

          # set the extent of the cartopy geoAxes to \"global\"
          #ax.set_global()
    
          # or alternatively, if you want to plot a certain region, use (example: Europe)
          #ax.set_extent([-180, 180, -63, 89], ccrs.PlateCarree())
          # ax.set_extent([-13, 43, 35, 70], ccrs.PlateCarree())
          # save the figure, adjusting the resolution 
    plt.legend()
    f.tight_layout()
    if mask==False:
        plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' + name+'_'+'seasonaltseries_'+model+'.png', dpi=300)
    else:
        plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' +'seasonaltseries_'+model+'.png', dpi=300)
    plt.show()

def plot_mon_or_tseries(var,case,outdir,cmor_table='Amon',model='cesm'):
    varname=var
    
    tseriesdir= outdir + '/'+case+'/' +cmor_table+'/' + varname +'/' 
    fn = varname + '_' + case.split('-')[0] +'_' + model + '_merge.nc'
    # open the dataset
    ds = xr.open_dataset(tseriesdir+fn)
    #ds_mask = xr.open_dataset(mask)
    da_scen=ds[varname]
    #da_mask=ds_mask['landmask']
    fn = varname + '_' + case.split('-')[1] +'_' + model + '_merge.nc'
    # open the dataset
    ds = xr.open_dataset(tseriesdir+fn)
    #ds_mask = xr.open_dataset(mask)
    da_ref=ds[varname]
    
    da_scen=da_scen.mean(dim=('lon','lat')).groupby('time.year').mean('time')  
    da_ref=da_ref.mean(dim=('lon','lat')).groupby('time.year').mean('time')  

    # title 
    title = da_ref.name

    # initiate the figure 

    f, ax = plt.subplots(1,1)
    xlims = (da_ref.year[0].values,da_ref.year[-1].values)
# plot data array as a map, with the previously defined argumnets
    da_scen.plot(ax=ax,xlim=xlims,label=case.split('-')[0])
    da_ref.plot(ax=ax,xlim=xlims,label=case.split('-')[1])

# plot the colorbar with all predefined arguments

    #cbar   = f.colorbar(h, ax=ax, cmap=cmap,spacing='uniform',orientation='horizontal',label = cbar_label,pad = 0.05,extend='both')

    # add the title

    ax.set_title(title)
    # adjust colorbar extent to axes extent

    ax.set_aspect('auto')

    # set the extent of the cartopy geoAxes to \"global\"
    #ax.set_global()

    # or alternatively, if you want to plot a certain region, use (example: Europe)
    #ax.set_extent([-180, 180, -63, 89], ccrs.PlateCarree())
    # ax.set_extent([-13, 43, 35, 70], ccrs.PlateCarree())
    # save the figure, adjusting the resolution 
    plt.legend()
    plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' + da_ref.name+'_'+'rawdata_tseries_'+model+'.png', dpi=300)
    
    plt.show()



def plot_mon_tseries(var,case,outdir,cmor_table='Amon',model='cesm'):
    varname=var
    var_local='absolute_'+var+'_local'
    var_nonlocal='absolute_'+var+'_nonlocal'
    var_total='absolute_'+var+'_total'
    
    tseriesdir= outdir + '/'+case+'/' +cmor_table+'/' + varname +'/' 
    fn = varname + '_' + case.split('-')[0] +'_' + model + '_signal-separated_absolute_values_'+case.split('-')[0]+ '-climate_merge.nc'
    fn1 = varname + '_' + case.split('-')[0] +'_' + model + '_signal-separated_absolute_values_'+case.split('-')[0]+ '-climate_000000.nc'
    fn2 = varname + '_' + case.split('-')[0] +'_' + model + '_signal-separated_absolute_values_'+case.split('-')[0]+ '-climate_000001.nc'
    fn3 = varname + '_' + case.split('-')[0] +'_' + model + '_signal-separated_absolute_values_'+case.split('-')[0]+ '-climate_000002.nc'
    fn4 = varname + '_' + case.split('-')[0] +'_' + model + '_signal-separated_absolute_values_'+case.split('-')[0]+ '-climate_000003.nc'
    fn5 = varname + '_' + case.split('-')[0] +'_' + model + '_signal-separated_absolute_values_'+case.split('-')[0]+ '-climate_000004.nc'
    
    f, ax = plt.subplots(3,1,sharex=True)
    ax = ax.flatten()
    k=0
    ens=[0,1,2,3,4,5]
    for fn in [fn1,fn2,fn3,fn4,fn5]:
        k=k+1
        # open the dataset
        ds = xr.open_dataset(tseriesdir+fn)
        #ds_mask = xr.open_dataset(mask)
        da_local=ds[var_local]
        da_nonlocal=ds[var_nonlocal]
        da_total=ds[var_total]
        #da_mask=ds_mask['landmask']
        i=0
        for da in [da_local, da_nonlocal, da_total]:
           i=i+1
           # annual means already taken
           da=da.mean(dim=('lon','lat')).rolling(time=12, center=True).mean()
  
           da['time']=np.arange(0,360,1)
           # title 
           title = da.name
    
           # initiate the figure 
    
           xlims = [0,361]
        # plot data array as a map, with the previously defined argumnets
           if i==1:
              h=da.plot(ax=ax[0],xlim=xlims,label=str(ens[k]))
                # add the title
              ax[0].set_title(title)
              ax[0].set_aspect('auto')
           elif i==2:
              h=da.plot(ax=ax[1],xlim=xlims,label=str(ens[k]))
                # add the title
              ax[1].set_title(title)
              ax[1].set_aspect('auto')
           elif i==3:
              h=da.plot(ax=ax[2],xlim=xlims,label=str(ens[k]))
                # add the title
              ax[2].set_title(title)
              ax[2].set_aspect('auto')

          # set the extent of the cartopy geoAxes to \"global\"
          #ax.set_global()
    
          # or alternatively, if you want to plot a certain region, use (example: Europe)
          #ax.set_extent([-180, 180, -63, 89], ccrs.PlateCarree())
          # ax.set_extent([-13, 43, 35, 70], ccrs.PlateCarree())
          # save the figure, adjusting the resolution 
    plt.legend()
    f.tight_layout()

    plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' + da.name+'_'+'seperatedtseries_'+model+'.png', dpi=300)
         
    plt.show()


def plot_fulltseries_mon(var,case,outdir,cmor_table='Amon',model='cesm'):
    varname=var
    var_local='absolute_'+var+'_local'
    var_nonlocal='absolute_'+var+'_nonlocal'
    var_total='absolute_'+var+'_total'
    
    tseriesdir= outdir + '/'+case+'/' +cmor_table+'/' + varname +'/' 
    fn = varname + '_' + case.split('-')[0] +'_' + model + '_signal-separated_absolute_values_'+case.split('-')[0]+ '-climate_merge.nc'
    # open the dataset
    ds = xr.open_dataset(tseriesdir+fn)
    #ds_mask = xr.open_dataset(mask)
    da_local=ds[var_local]
    da_nonlocal=ds[var_nonlocal]
    da_total=ds[var_total]
    #da_mask=ds_mask['landmask']

    for da in [da_local, da_nonlocal, da_total]:
        # annual means already taken
     
          da=da.mean(dim=('lon','lat')).groupby('time.year').mean('time')  
    
          # title 
          title = da.name
    
          # initiate the figure 
    
          f, ax = plt.subplots(1,1)
          xlims = (da.year[0].values,da.year[-1].values)
        # plot data array as a map, with the previously defined argumnets
          da.plot(ax=ax,xlim=xlims)
        # plot the colorbar with all predefined arguments

          #cbar   = f.colorbar(h, ax=ax, cmap=cmap,spacing='uniform',orientation='horizontal',label = cbar_label,pad = 0.05,extend='both')
    
          # add the title
    
          ax.set_title(title)
          # adjust colorbar extent to axes extent
    
          ax.set_aspect('auto')
    
          # set the extent of the cartopy geoAxes to \"global\"
          #ax.set_global()
    
          # or alternatively, if you want to plot a certain region, use (example: Europe)
          #ax.set_extent([-180, 180, -63, 89], ccrs.PlateCarree())
          # ax.set_extent([-13, 43, 35, 70], ccrs.PlateCarree())
          # save the figure, adjusting the resolution 
          plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' + da.name+'_'+'fulltseries_'+model+'.png', dpi=300)
         
          plt.show()


def plot_hatchedmean(var,case,outdir,vlims=False,season=False,cmor_table='Amon',model='cesm',mask=False,relative=False,level='none'):
    varname=var
    var_local=var+'_local'
    var_nonlocal=var+'_nonlocal'
    var_total=var+'_total'
    
    tseriesdir= outdir + '/'+case+'/' +cmor_table+'/' + varname +'/' 
    
    if season== False & os.path.isfile(tseriesdir + varname + '_' + case +'_' + model + '_ensmean_signal-separated.nc'):
        fn1 = varname + '_' + case +'_' + model + '_timmean000000_signal-separated.nc'
        fn2 = varname + '_' + case +'_' + model + '_timmean000001_signal-separated.nc'
        fn3 = varname + '_' + case +'_' + model + '_timmean000002_signal-separated.nc'
        fn4 = varname + '_' + case +'_' + model + '_timmean000003_signal-separated.nc'
        fn5 = varname + '_' + case +'_' + model + '_timmean000004_signal-separated.nc'
        if model =='cesm':
           sign_local=np.zeros((192,288))
           sign_nonlocal=np.zeros((192,288))
           sign_total=np.zeros((192,288))
        elif model=='mpiesm':
           sign_local=np.zeros((96,192))
           sign_nonlocal=np.zeros((96,192))
           sign_total=np.zeros((96,192))

        for fn in [fn1,fn2,fn3,fn4,fn5]:
             ds = xr.open_dataset(tseriesdir+fn)
             #ds_mask = xr.open_dataset(mask)
             if level != 'none':
                da_local=ds[var_local].mean('time')[level,:,:]
                da_nonlocal=ds[var_nonlocal].mean('time')[level,:,:]
                da_total=ds[var_total].mean('time')[level,:,:]
             else:
                da_local=ds[var_local].mean('time')
                da_nonlocal=ds[var_nonlocal].mean('time')
                da_total=ds[var_total].mean('time')
             
             if fn==fn1:
                da_local_check=da_local
                da_nonlocal_check=da_nonlocal
                da_total_check=da_total
             else:
                sign_local=sign_local+((da_local_check[:,:]> 0) == (da_local[:,:]>0))
                sign_nonlocal=sign_nonlocal+((da_nonlocal_check[:,:]> 0) == (da_nonlocal[:,:]>0))
                sign_total=sign_total+((da_total_check[:,:]> 0) == (da_total[:,:]>0))

        #check where val is 4 -> all equal signs give val 1 else give val 0
        fn = varname + '_' + case +'_' + model + '_ensmean_signal-separated.nc'
        # open the dataset
        ds = xr.open_dataset(tseriesdir+fn)
        #ds_mask = xr.open_dataset(mask)
        if level != 'none':
            da_local=ds[var_local].mean('time')[level,:,:]
            da_nonlocal=ds[var_nonlocal].mean('time')[level,:,:]
            da_total=ds[var_total].mean('time')[level,:,:]
        else:
            da_local=ds[var_local].mean('time')
            da_nonlocal=ds[var_nonlocal].mean('time')
            da_total=ds[var_total].mean('time')
                
        #da_mask=ds_mask['landmask']

            
    elif season!=False & os.path.isfile(tseriesdir + varname + '_' + case +'_' + model + '_' +season+'ensmean_signal-separated.nc'):
        fn = varname + '_' + case +'_' + model + '_' +season+'ensmean_signal-separated.nc'
        
        fn1 = varname + '_' + case +'_' + model + '_'+season+'mean000000_signal-separated.nc'
        fn2 = varname + '_' + case +'_' + model + '_'+season+'mean000001_signal-separated.nc'
        fn3 = varname + '_' + case +'_' + model + '_'+season+'mean000002_signal-separated.nc'
        fn4 = varname + '_' + case +'_' + model + '_'+season+'mean000003_signal-separated.nc'
        fn5 = varname + '_' + case +'_' + model + '_'+season+'mean000004_signal-separated.nc'
        if model =='cesm':
           sign_local=np.zeros((192,288))
           sign_nonlocal=np.zeros((192,288))
           sign_total=np.zeros((192,288))
        elif model=='mpiesm':
           sign_local=np.zeros((96,192))
           sign_nonlocal=np.zeros((96,192))
           sign_total=np.zeros((96,192))

        for fn in [fn1,fn2,fn3,fn4,fn5]:
             ds = xr.open_dataset(tseriesdir+fn)
             #ds_mask = xr.open_dataset(mask)
             da_local=ds[var_local].mean('time')
             da_nonlocal=ds[var_nonlocal].mean('time')
             da_total=ds[var_total].mean('time')
             if fn==fn1:
                 da_local_check=da_local
                 da_nonlocal_check=da_nonlocal
                 da_total_check=da_total
             else:
                 sign_local=sign_local+((da_local_check[:,:]> 0) == (da_local[:,:]>0))
                 sign_nonlocal=sign_nonlocal+((da_nonlocal_check[:,:]> 0) == (da_nonlocal[:,:]>0))
                 sign_total=sign_total+((da_total_check[:,:]> 0) == (da_total[:,:]>0))

        #check where val is 4 -> all equal signs give val 1 else give val 0
        fn = varname + '_' + case +'_' + model + '_ensmean_signal-separated.nc'
        # open the dataset
        ds = xr.open_dataset(tseriesdir+fn)
        #ds_mask = xr.open_dataset(mask)
        da_local=ds[var_local]
        da_nonlocal=ds[var_nonlocal]
        da_total=ds[var_total]
        #da_mask=ds_mask['landmask']

    else:
        print('unknow issue, missing data or wrong input!!')
    i=0
    for da in [da_local, da_nonlocal, da_total]:
          name = da.name
          print(da)
          if var=='PRECC' or var=='PRECT':
              if relative==True:
                ds_ctl= xr.open_dataset(outdir+'/frst-ctl/Amon/PRECC/'+ 'PRECC_ctl_cesm_timmean.nc')
                da_ctl=ds_ctl[var]
                da_brol=da.values/da_ctl.values
                da.values=da_brol*100
              else:
                da=da*86400000
                print('normal')
              cmap='RdBu'
          elif var=='pr':
              if relative==True:
                ds_ctl= xr.open_dataset(outdir +'/crop-ctl/'+cmor_table+'/'+var+'/'+ 'pr_ctl_mpiesm_150-years.nc')
                da_ctl=ds_ctl[var].mean('time')
                da_brol=da.values/da_ctl.values
                da.values=da_brol*100
              else:
                da=da*86400
              cmap='RdBu'
          else:
              cmap = 'coolwarm'
        # annual means already taken
          i=i+1
    #     da_mean=xr.where(da_mask==1,da_mean,0)
    
          # title 
          da=mask_data(da,case,model,mask)

          # define upper and lower plotting limits (by default min and max of dataarray)
          if vlims==False:
              plot_lims = [da.min(), da.max()]
          else:
              plot_lims = [vlims[0],vlims[1]]
    
    # define colormap (more info on colormaps: https://matplotlib.org/users/colormaps.html)
    
          #cmap, norm = mpu.from_levels_and_cmap(levels, 'RdBu_r', extend='both')
          cbar_label = varname
    
    ############ define the projection,
    
          projection = ccrs.PlateCarree()
    
          # initiate the figure 
    
          f, ax = plt.subplots(1,1,subplot_kw={'projection':projection})
    
           # add the coastlines to the plot

          ax.coastlines()
        # plot data array as a map, with the previously defined argumnets

          h=da.plot(ax=ax, cmap=cmap, vmin=plot_lims[0], vmax=plot_lims[1], add_colorbar=False)
        # plot the colorbar with all predefined arguments

          cbar   = f.colorbar(h, ax=ax, cmap=cmap,spacing='uniform',orientation='horizontal',label = cbar_label,pad = 0.05,extend='both')
    
          # add the title
    
          ax.set_title(name)
          # adjust colorbar extent to axes extent
    
          ax.set_aspect('auto')
    
          # set the extent of the cartopy geoAxes to \"global\"
          #ax.set_global()
    
          # or alternatively, if you want to plot a certain region, use (example: Europe)
          ax.set_extent([-180, 180, -63, 89], ccrs.PlateCarree())
          
          levels = [0, 3.2, 5]
          hatches = ['', '...']
          if i==1:
              if mask==False:
                  sign_local=mask_data(sign_local,case,model,'lnd')
              else:
                  sign_local=mask_data(sign_local,case,model,mask)
              ax.contourf(da.lon, da.lat, sign_local, levels=levels, hatches=hatches, colors='none')
          elif i==2:
              sign_nonlocal=mask_data(sign_nonlocal,case,model,mask)
              ax.contourf(da.lon, da.lat, sign_nonlocal, levels=levels, hatches=hatches, colors='none')
          elif i==3:
              sign_total=mask_data(sign_total,case,model,mask)
              ax.contourf(da.lon, da.lat, sign_total, levels=levels, hatches=hatches, colors='none')
          # ax.set_extent([-13, 43, 35, 70], ccrs.PlateCarree())
          # save the figure, adjusting the resolution 
          if level!='none':
              plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' + name+'_'+'hatchedmean_level'+str(level)+'_'+model+'.png', dpi=300)
          if mask== False:
            if season ==False:
                plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' + name+'_'+'hatchedmean_'+model+'.png', dpi=300)
            else:
                plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' + name+'_'+season+'_hatchedmean_'+model+'.png', dpi=300)
          else:
            if season ==False:
                plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' + name+'_'+'hatchedmean_'+mask+'_'+model+'.png', dpi=300)
            else:
                plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' + name+'_'+season+'_hatchedmean_'+mask+'_'+model+'.png', dpi=300)
          plt.show()

def plot_mean(var,case,outdir,vlims=False,season=False,cmor_table='Amon',model='cesm',mask=False):
    varname=var
    var_local=var+'_local'
    var_nonlocal=var+'_nonlocal'
    var_total=var+'_total'
    
    tseriesdir= outdir + '/'+case+'/' +cmor_table+'/' + varname +'/' 
    if season== False & os.path.isfile(tseriesdir + varname + '_' + case +'_' + model + '_ensmean_signal-separated.nc'):
        fn = varname + '_' + case +'_' + model + '_ensmean_signal-separated.nc'
        # open the dataset
        ds = xr.open_dataset(tseriesdir+fn)
        #ds_mask = xr.open_dataset(mask)
        da_local=ds[var_local]
        da_nonlocal=ds[var_nonlocal]
        da_total=ds[var_total]
        #da_mask=ds_mask['landmask']
    elif season!=False & os.path.isfile(tseriesdir + varname + '_' + case +'_' + model + '_' +season+'ensmean_signal-separated.nc'):
        fn = varname + '_' + case +'_' + model + '_' +season+'ensmean_signal-separated.nc'
        # open the dataset
        ds = xr.open_dataset(tseriesdir+fn)
        #ds_mask = xr.open_dataset(mask)
        da_local=ds[var_local]
        da_nonlocal=ds[var_nonlocal]
        da_total=ds[var_total]
        #da_mask=ds_mask['landmask']
    else:
        print('unknow issue, missing data or wrong input!!')
    
    for da in [da_local, da_nonlocal, da_total]:
        # annual means already taken
          name=da.name
          da=mask_data(da,case,model,mask)
          # title 
          title = da.name
          # define upper and lower plotting limits (by default min and max of dataarray)
          if vlims==False:
              plot_lims = [da.min(), da.max()]
          else:
              plot_lims = [vlims[0],vlims[1]]
    
    # define colormap (more info on colormaps: https://matplotlib.org/users/colormaps.html)
    
          #cmap, norm = mpu.from_levels_and_cmap(levels, 'RdBu_r', extend='both')
          cmap = 'coolwarm'# define colorbar label (including unit!)
          cbar_label = '?'
    
    ############ define the projection,
    
          projection = ccrs.PlateCarree()
    
          # initiate the figure 
    
          f, ax = plt.subplots(1,1,subplot_kw={'projection':projection})
    
           # add the coastlines to the plot

          ax.coastlines()
        # plot data array as a map, with the previously defined argumnets

          h=da.plot(ax=ax, cmap=cmap, vmin=plot_lims[0], vmax=plot_lims[1], add_colorbar=False)
        # plot the colorbar with all predefined arguments

          cbar   = f.colorbar(h, ax=ax, cmap=cmap,spacing='uniform',orientation='horizontal',label = cbar_label,pad = 0.05,extend='both')
    
          # add the title
    
          ax.set_title(title)
          # adjust colorbar extent to axes extent
    
          ax.set_aspect('auto')
    
          # set the extent of the cartopy geoAxes to \"global\"
          #ax.set_global()
    
          # or alternatively, if you want to plot a certain region, use (example: Europe)
          ax.set_extent([-180, 180, -63, 89], ccrs.PlateCarree())
          # ax.set_extent([-13, 43, 35, 70], ccrs.PlateCarree())
          # save the figure, adjusting the resolution 
          if mask ==False:
            if season ==False:
                plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' + name+'_'+'mean_'+model+'.png', dpi=300)
            else:
                plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+case+ '_' + name+'_'+season+'_mean_'+model+'.png', dpi=300)
          else:
            if season ==False:
                plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' + name+'_'+'mean_'+mask+'_'+model+'.png', dpi=300)
            else:
                plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+case+ '_' + name+'_'+season+'_mean_'+mask+'_'+model+'.png', dpi=300)
          plt.show()



def plot_latlon(var,case,outdir,vlims=False,season=False,cmor_table='Amon',model='cesm',mask=False):
    varname=var
    var_local=var+'_local'
    var_nonlocal=var+'_nonlocal'
    var_total=var+'_total'
    
    tseriesdir= outdir + '/'+case+'/' +cmor_table+'/' + varname +'/' 
    if season== False & os.path.isfile(tseriesdir + varname + '_' + case +'_' + model + '_ensmean_signal-separated.nc'):
        fn = varname + '_' + case +'_' + model + '_ensmean_signal-separated.nc'
        # open the dataset
        ds = xr.open_dataset(tseriesdir+fn)
        #ds_mask = xr.open_dataset(mask)
        da_local=ds[var_local]
        da_nonlocal=ds[var_nonlocal]
        da_total=ds[var_total]
        #da_mask=ds_mask['landmask']
    elif season!=False & os.path.isfile(tseriesdir + varname + '_' + case +'_' + model + '_' +season+'ensmean_signal-separated.nc'):
        fn = varname + '_' + case +'_' + model + '_' +season+'ensmean_signal-separated.nc'
        # open the dataset
        ds = xr.open_dataset(tseriesdir+fn)
        #ds_mask = xr.open_dataset(mask)
        da_local=ds[var_local]
        da_nonlocal=ds[var_nonlocal]
        da_total=ds[var_total]
        #da_mask=ds_mask['landmask']
    else:
        print('unknow issue, missing data or wrong input!!')
    
    for da in [da_local, da_nonlocal, da_total]:
        # annual means already taken
          name=da.name
          da=mask_data(da,case,model,mask)
          # title 
          title = da.name
          # define upper and lower plotting limits (by default min and max of dataarray)
          if vlims==False:
              plot_lims = [da.min(), da.max()]
          else:
              plot_lims = [vlims[0],vlims[1]]
    
    # define colormap (more info on colormaps: https://matplotlib.org/users/colormaps.html)
    
          #cmap, norm = mpu.from_levels_and_cmap(levels, 'RdBu_r', extend='both')
          cmap = 'coolwarm'# define colorbar label (including unit!)
          cbar_label = '?'
    
    ############ define the projection,
    
          projection = ccrs.PlateCarree()
    
          # initiate the figure 
    
          #f, ax = plt.subplots(2,2)
          fig = plt.figure()
          gs = fig.add_gridspec(3,3)

          ax1 = fig.add_subplot(gs[1:3, 0:2], projection=ccrs.PlateCarree())
          ax1.set_extent([-180, 180, -63, 89], crs=ccrs.PlateCarree())
          ax1.coastlines(resolution='auto', color='k')
          #ax1.gridlines(color='lightgrey', linestyle='-', draw_labels=True)

          ax2 = fig.add_subplot(gs[0, 0:2])
          #da.mean(dim='lat').plot(ax=ax2)
          dalon=da.mean(dim='lat')
          if model =='cesm':
            LON=np.arange(-180,180,1.25)
            values_shifted=np.zeros(288)
            values_shifted[0:144]=dalon.mean('time')[144:]
            values_shifted[144:]=dalon.mean('time')[0:144]
          elif model=='mpiesm':
            #LON=np.zeros_like(da_lon[:,0])
            LON=np.arange(-180,180,1.875)
            values_shifted=np.zeros(192)
            values_shifted[0:96]=dalon.mean('time')[96:]
            values_shifted[96:]=dalon.mean('time')[0:96]
          ax2.plot(LON,values_shifted)
          #base = plt.gca().transData
          #rot = transforms.Affine2D().rotate_deg(90)
          ax3 = fig.add_subplot(gs[1:3, 2])
#          da.mean(dim='lon').plot(ax=ax3)
          dalat=da.mean(dim='lon')
          ax3.plot(dalat.mean('time')[:],dalat.lat)         
 # add the coastlines to the plot
          #ax[1,0] = plt.axes(projection=projection)

          #ax[1,0].coastlines()
        # plot data array as a map, with the previously defined argumnets

          h=da.plot(ax=ax1, cmap=cmap, vmin=plot_lims[0], vmax=plot_lims[1], add_colorbar=False)
        # plot the colorbar with all predefined arguments
          if vlims==False:
              cbar   = fig.colorbar(h, ax=ax1, cmap=cmap,spacing='uniform',orientation='horizontal',label = cbar_label,pad = 0.05)
          else:
              cbar   = fig.colorbar(h, ax=ax1, cmap=cmap,spacing='uniform',orientation='horizontal',label = cbar_label,pad = 0.05,extend='both')
    
          # add the title
    
          ax1.set_title(title)
          
    
          # set the extent of the cartopy geoAxes to \"global\"
          #ax.set_global()
    
          # or alternatively, if you want to plot a certain region, use (example: Europe)
          #ax[1,0].set_extent([-180, 180, -63, 89], ccrs.PlateCarree())
          # ax.set_extent([-13, 43, 35, 70], ccrs.PlateCarree())
          # save the figure, adjusting the resolution 
          #plt.tight_layout()
          if mask ==False:
            if season ==False:
                plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' + name+'_'+'latlonplot_'+model+'.png', dpi=300)
            else:
                plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+case+ '_' + name+'_'+season+'latlonplot_'+model+'.png', dpi=300)
          else:
            if season ==False:
                plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+ case+ '_' + name+'_'+'latlonplot_'+mask+'_'+model+'.png', dpi=300)
            else:
                plt.savefig(outdir + '/'+case+'/' +cmor_table+'/'+'plots'+ '/'+case+ '_' + name+'_'+season+'latlonplot_'+mask+'_'+model+'.png', dpi=300)
          plt.show()


def mask_data(da,case,model,mask):
     # annual means already taken
    import sys
    import os
    import string
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import netCDF4 as netcdf4
    import xarray as xr
    if model=='ecearth':
        if mask== 'lnd':
            fn_mask='/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/lamaclima_wp1/LU_maps/interpolated_ec-earth_land_sea_mask.nc'
            ds_mask = xr.open_dataset(fn_mask)
            da_mask=ds_mask['mask']
            da, da_mask = xr.align(da, da_mask, join="override")
            da=xr.where(da_mask==1,da,np.nan)
    if model == 'cesm':
        if mask== 'lnd':
            fn_mask='/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/lamaclima_wp1/LU_maps/landmask_cesm.nc'
            ds_mask = xr.open_dataset(fn_mask)
            da_mask=ds_mask['landmask']
            da, da_mask = xr.align(da, da_mask, join="override")
            da=xr.where(da_mask==1,da,np.nan)
        elif mask== 'lcc_crop' :
            fn_mask='/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/lamaclima_wp1/LU_maps/masks/'+case+'_mask.nc'
            ds_mask = xr.open_dataset(fn_mask)
            da_mask=ds_mask['CROP'].values
            da_mask_2=xr.DataArray(da_mask,coords=(da.lat,da.lon))
            #da, da_mask = xr.align(da, da_mask, join="override")
            da=xr.where(da_mask_2==1,da,np.nan)

        elif mask== 'lcc_frst':
            fn_mask='/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/lamaclima_wp1/LU_maps/masks/'+case+'_mask.nc'
            ds_mask = xr.open_dataset(fn_mask)
            da_mask=ds_mask['FRST'].values
            da_mask_2=xr.DataArray(da_mask,coords=(da.lat,da.lon))
            #da, da_mask = xr.align(da, da_mask, join="override")
            da=xr.where(da_mask_2==1,da,np.nan)
        elif mask== 'lcc_crop50':
            fn_mask='/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/lamaclima_wp1/LU_maps/masks/'+case+'50_mask.nc'
            ds_mask = xr.open_dataset(fn_mask)
            da_mask=ds_mask['CROP'].values
            da_mask_2=xr.DataArray(da_mask,coords=(da.lat,da.lon))
            #da, da_mask = xr.align(da, da_mask, join="override")
            da=xr.where(da_mask_2==1,da,np.nan)

        elif mask== 'lcc_frst50':
            fn_mask='/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/lamaclima_wp1/LU_maps/masks/'+case+'50_mask.nc'
            ds_mask = xr.open_dataset(fn_mask)
            da_mask=ds_mask['FRST'].values
            da_mask_2=xr.DataArray(da_mask,coords=(da.lat,da.lon))
            #da, da_mask = xr.align(da, da_mask, join="override")
            da=xr.where(da_mask_2==1,da,np.nan)
        elif (mask=='boreal' or mask=='southern' or mask=='intermediate' or mask=='subtropics_NH' or mask=='subtropics_SH' or mask=='tropical') and mask!=False:
            fn_mask='/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/lamaclima_wp1/LU_maps/masks/'+mask+'_mask.nc'
            ds_mask = xr.open_dataset(fn_mask)
            print(ds_mask)
            da_mask=ds_mask['latmask']
            print(da_mask)
            da_mask_2=xr.DataArray(da_mask,coords=(da.lat,da.lon))
            print(da_mask_2)
            print(da)
            da=xr.where(da_mask_2==1,da,np.nan)
    elif model=='mpiesm':
        if mask== 'lnd':
            fn_mask='/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/lamaclima_wp1/LU_maps/jsbach_T63GR15_11tiles_5layers_2005_dynveg_slm_glac.nc'
            ds_mask = xr.open_dataset(fn_mask)
            da_mask=ds_mask['slm'].values
            da_mask=np.flipud(da_mask)
            da_mask_2=xr.DataArray(da_mask,coords=(da.lat,da.lon))
            #da, da_mask = xr.align(da, da_mask, join="override")
            da=xr.where(da_mask_2==1,da,np.nan)
    return da



def mask_lcc(case='crop-ctl',intensity=False):
    fn_CTL = 'surfdata_0.9x1.25_hist_78pfts_CMIP6_simyr2014_noWH.nc'
    fn_CROP = 'surfdata_0.9x1.25_hist_78pfts_CMIP6_simyr2014_CROP_idealised.nc'
    fn_FRST = 'surfdata_0.9x1.25_hist_78pfts_CMIP6_simyr2014_FRST_idealised.nc'
    fn_IRR = 'surfdata_0.9x1.25_hist_78pfts_CMIP6_simyr2014_IRR_idealised.nc'
    outdir = '/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/lamaclima_wp1/LU_maps/'
    procdir=outdir
    os.chdir(procdir)

    #fn_list=[fn_CTL,fn_CROP,fn_FRST,fn_IRR,fn_iCROP,fn_iFRST,fn_iIRR]
    if case=='crop-ctl':
        fn_list=[fn_CTL,fn_CROP]
        diff_list1=['','']
        diff_list2=['','']
    i=0
    for fn in fn_list:
        i=i+1
        if not os.path.isfile(outdir + fn):
            print(fn + ' does not exists in ')
            print(work)
        else: 
            print('kak2')
            # open the dataset
            ds = xr.open_dataset(outdir+fn)
            da_crop=ds['PCT_CROP']
            da_natveg=ds['PCT_NATVEG']
            da_natpft=ds['PCT_NAT_PFT']
            da_forest=da_natveg[:]-(da_natveg[:]/(100)*(da_natpft[0,:,:]+np.sum(da_natpft[12:,:,:],axis=0))/(100))*100
            # title 
            da=da_crop
            title = da.name
            print(da.name)
            ds_new = xr.Dataset({"PCT_CROP": (("lsmlat", "lsmlon"), da_crop),"PCT_NATVEG": (("lsmlat", "lsmlon"), da_natveg),"PCT_FRST": (("lsmlat", "lsmlon"), da_forest)},coords={"lsmlat": ds.lsmlat,"lsmlon": ds.lsmlon})
            #ds_new = xr.Dataset({"PCT_CROP": (("lat", "lon"), da_crop)},coords={"lat":np.arange(-90,90,0.9375),"lon": np.arange(-180,180,1.25)})
            ds_new.to_netcdf('/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/lamaclima_wp1/LU_maps/masks/'+fn.split('_')[-2]+"_mask.nc")
            print(ds_new)

            if case =='crop-ctl':
                diff_list1[i-1]=da_crop
                diff_list2[i-1]=da_forest

    if case =='crop-ctl':
        print('kak5')

        diff1=diff_list1[1]-diff_list1[0]
        diff2=diff_list2[0]-diff_list2[1]
        if intensity==False:
            mask_diff1=xr.where(diff1>1,1,np.nan)
            mask_diff2=xr.where(diff2>1,1,np.nan)
            ds_crop_diff = xr.Dataset({"CROP": (("lsmlat", "lsmlon"), mask_diff1),"FRST": (("lsmlat", "lsmlon"), mask_diff2)},coords={"lsmlat": ds.lsmlat,"lsmlon": ds.lsmlon})
            ds_crop_diff.to_netcdf('/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/lamaclima_wp1/LU_maps/masks/crop-ctl_mask.nc')
        else:
            mask_diff1=xr.where(diff1>intensity,1,np.nan)
            mask_diff2=xr.where(diff2>intensity,1,np.nan)
            ds_crop_diff = xr.Dataset({"CROP": (("lsmlat", "lsmlon"), mask_diff1),"FRST": (("lsmlat", "lsmlon"), mask_diff2)},coords={"lsmlat": ds.lsmlat,"lsmlon": ds.lsmlon})
            ds_crop_diff.to_netcdf('/scratch/leuven/projects/lt1_2020_es_pilot/project_output/bclimate/sdeherto/lamaclima_wp1/LU_maps/masks/crop-ctl'+str(intensity)+'_mask.nc')


