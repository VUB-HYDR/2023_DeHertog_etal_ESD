#! /usr/bin/python

import numpy as np 
from netCDF4 import Dataset

#Similar to MIssissipi case, I included area_mask with the cell areas of the mesh in CESM for the new 
#A_gridcell 

def getconstants_pressure_CESM(latnrs,lonnrs,invariant_data,area_mask): 
	# def getconstants in Python is the same as function in MATLAB. 
    
    # load the latitude and longitude from the invariants file
    latitude = Dataset(invariant_data, mode = 'r').variables['lat'][:] # [degrees north]
    longitude = Dataset(invariant_data, mode = 'r').variables['lon'][:] # [degrees east]

    # Create land-sea-mask (in this model lakes are considered part of the land)
    lsm = Dataset(invariant_data, mode = 'r').variables['landmask'][:,:] # 0 = sea, 1 = land

    #for n in range(len(lake_mask[:,0])): # 1 = sea, 0 = land
    #    lsm[lake_mask[n,0],lake_mask[n,1]] = 0

#    lsm[0,:] = 0 # the northern boundary is always oceanic = 0
#    lsm[-1,:] = 0 # the southern boundary is always oceanic = 0
    
    # Constants 
    g = 9.80665 # [m/s2] from ERA-interim archive
    density_water = 1000 # [kg/m3]
    dg = 111089.56 # [m] length of 1 degree latitude
    if model =='cesm':
      timestep = 6*3600 # [s] timestep in the ERA-interim archive 
    else:
      timestep = 3*3600 # [s] timestep in the ERA-interim archive 
    Erad = 6.371e6 # [m] Earth radius
    
    # Semiconstants
    gridcell = np.abs(longitude[1] - longitude[0]) # [degrees] grid cell size
    
    # new area size calculation:
    lat_n_bound = np.minimum(90.0 , latitude + 0.5*gridcell)
    lat_s_bound = np.maximum(-90.0 , latitude - 0.5*gridcell)
    
    A_gridcell = np.zeros([len(latitude),1])
    A_gridcell[:,0] = (np.pi/180.0)*Erad**2 * abs( np.sin(lat_s_bound*np.pi/180.0) - np.sin(lat_n_bound*np.pi/180.0) ) * gridcell
    A_gridcell[:,0] = Dataset(area_mask, mode = 'r').variables['cell_area'][:,0] #
    
    L_N_gridcell = gridcell * np.cos((latitude + gridcell / 2.0) * np.pi / 180.0) * dg # [m] length northern boundary of a cell
    L_S_gridcell = gridcell * np.cos((latitude - gridcell / 2.0) * np.pi / 180.0) * dg # [m] length southern boundary of a cell
    L_EW_gridcell = gridcell * dg # [m] length eastern/western boundary of a cell 
    
    return latitude , longitude , lsm , g , density_water , timestep , A_gridcell , L_N_gridcell , L_S_gridcell , L_EW_gridcell , gridcell
