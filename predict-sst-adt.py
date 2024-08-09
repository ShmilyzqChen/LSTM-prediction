# -*- coding: utf-8 -*-
"""
Created on Thu May  9 22:26:46 2024

@author: Zhiqiang Chen of Hohai University
"""
from keras.models import load_model
import os
from datetime import datetime, timedelta
import xarray as xr
import dask
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import scipy.io
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import fnmatch
import time as tim
from scipy.io import savemat
import datetime as dt
# from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings("ignore")

def calculate_NSE(data1,data2):
    # Calculate Nash-Sutcliffe Efficiency (NSE), equal to function r2score
    # The NSE is closer to 1, the better
    numerator = np.sum(np.power(data1-data2,2))
    denominator = np.sum(np.power(data1-np.mean(data1),2))
    nse = 1 - numerator/denominator
    return nse

def data_standardization(data):
    # Data normalization (MinMaxScaler)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler

def data_anti_standardization(data_normalized,scaler):
    # Inverse normalization
    data_original = scaler.inverse_transform(data_normalized.reshape(1, -1))
    return data_original

def get_doys_predict(target_date,days_predict):
    # Get day of year for prediction dates
    doys_predict = []
    date_predict = np.ones([days_predict,1],dtype=int)
    for day in range(1,days_predict + 1):
        target_day = target_date + timedelta(days=day)        
        doy = target_day.timetuple().tm_yday - 1
        # date_predict.append(int(target_day.strftime("%Y%m%d")))
        date_predict[day-1] = int(target_day.strftime("%Y%m%d"))
        doys_predict.append(doy)
    return doys_predict,date_predict

def calculate_statistics_4_evaluate(days_predict,y_true,y_predict):
    # Calculate MAE, RMSE, NSE, R^2 for evaluation
    # 0-mae 1-rmse 2-r2score or named Nash-Sutcliffe Efficiency  3-P-correlation
    statistics_4_out = np.full([y_true.shape[0], 4], np.nan)    
    for k_step in range(y_true.shape[0]):
        data1 = y_true[k_step].flatten()
        data2 = y_predict[k_step].flatten()
        valid_indices = ~np.isnan(data1) & ~np.isnan(data2)
        data1 = data1[valid_indices]
        data2 = data2[valid_indices]
        statistics_4_out[k_step, 0] = mean_absolute_error(data1, data2)
        statistics_4_out[k_step, 1] = np.sqrt(mean_squared_error(data1, data2))
        statistics_4_out[k_step, 2] = r2_score(data1, data2)
        statistics_4_out[k_step, 3] = np.corrcoef(data1, data2)[0, 1]
    return statistics_4_out

def read_ncfiles_data_before_date(var, data_in_path, days_history, target_date):
    # Read data from netCDF files before the target date
    if var.lower() == 'sst':
        data_in_path_SST = data_in_path
        SST_head_name = 'oisst-avhrr-v02r01.'
        nc_files = []
        doys_history = []
        for day in range(days_history, -1, -1):
            target_day = target_date - timedelta(days=day)
            mm = target_day.strftime("%Y%m")
            ymd = target_day.strftime("%Y%m%d")
            SST_file = f'{data_in_path_SST}{mm}/{SST_head_name}{ymd}.nc'
            doy = target_day.timetuple().tm_yday
            if os.path.exists(SST_file):
                nc_files.append(SST_file)
                doys_history.append(doy)
            else:
                raise ValueError(f'{SST_file} does not exist, please add it in the target directory')
    elif var.lower() == 'adt':
        data_in_path_SSH = data_in_path
        SSH_head_name = 'dt_global_allsat_phy_l4_'
        nc_files = []
        doys_history = []
        for day in range(days_history, -1, -1):
            target_day = target_date - timedelta(days=day)
            yy = target_day.strftime("%Y")
            ymd = target_day.strftime("%Y%m%d")
            doy = target_day.timetuple().tm_yday
            path1 = f'{data_in_path_SSH}/{yy}'
            SSH_file_pattern = f'{SSH_head_name}{ymd}_*.nc'
            found = False
            for f_name in os.listdir(path1):
                if fnmatch.fnmatch(f_name, SSH_file_pattern):
                    SSH_file = f'{path1}/{f_name}'
                    nc_files.append(SSH_file)
                    doys_history.append(doy)
                    found = True
                    break
            if not found:
                raise ValueError(f'{SSH_file_pattern} does not exist, please add it in the target directory')
    else:
        raise IndexError('The input var name is not considered right now')
        
    # Read and return data from netCDF files
    ds = xr.open_mfdataset(nc_files)
    if var.lower() == 'sst': 
        lon0 = ds['lon'].values
        lat0 = ds['lat'].values
    elif var.lower() == 'adt':
        lon0 = ds['longitude'].values
        lat0 = ds['latitude'].values
    else:
        raise AttributeError
    # lonlat for NIP
    indlon = np.array([np.where(lon0 == 99.875), np.where(lon0 == 150.125)]).flatten()
    lon0 = lon0[indlon[0]:indlon[1]+1]
    indlat = np.array([np.where(lat0 == -1.125), np.where(lat0 == 50.125)]).flatten()
    lat0 = lat0[indlat[0]:indlat[1]+1]
    data_history = np.squeeze(ds[var]).values
    data_history = data_history[:, indlat[0]:indlat[1]+1, indlon[0]:indlon[1]+1]
    return lon0,lat0,doys_history,data_history

def read_ncfiles_data_after_date(var, data_in_path, days_predict, target_date):
    # Read data from netCDF files after the target date   
    if var.lower() == 'sst':
        data_in_path_SST = data_in_path
        SST_head_name = 'oisst-avhrr-v02r01.'
        nc_files = []
        doys_predict = []
        for day in range(1,days_predict+1):
            target_day = target_date + timedelta(days=day)
            mm = target_day.strftime("%Y%m")
            ymd = target_day.strftime("%Y%m%d")
            SST_file = f'{data_in_path_SST}{mm}/{SST_head_name}{ymd}.nc'
            doy = target_day.timetuple().tm_yday
            if os.path.exists(SST_file):
                nc_files.append(SST_file)
                doys_predict.append(doy)
            else:
                raise ValueError(f'{SST_file} does not exist, please add it in the target directory')
    elif var.lower() == 'adt':
        data_in_path_SSH = data_in_path
        SSH_head_name = 'dt_global_allsat_phy_l4_'
        nc_files = []
        doys_predict = []
        for day in range(1,days_predict+1):
            target_day = target_date + timedelta(days=day)
            yy = target_day.strftime("%Y")
            ymd = target_day.strftime("%Y%m%d")
            doy = target_day.timetuple().tm_yday
            path1 = f'{data_in_path_SSH}/{yy}'
            SSH_file_pattern = f'{SSH_head_name}{ymd}_*.nc'
            found = False
            for f_name in os.listdir(path1):
                if fnmatch.fnmatch(f_name, SSH_file_pattern):
                    SSH_file = f'{path1}/{f_name}'
                    nc_files.append(SSH_file)
                    doys_predict.append(doy)
                    found = True
                    break
            if not found:
                raise ValueError(f'{SSH_file_pattern} does not exist, please add it in the target directory')
    else:
        raise IndexError('The input var name is not considered right now')
        
    ## read the proper ncfiles
    ds = xr.open_mfdataset(nc_files)
    if var.lower() == 'sst': 
        lon0 = ds['lon'].values
        lat0 = ds['lat'].values
    elif var.lower() == 'adt':
        lon0 = ds['longitude'].values
        lat0 = ds['latitude'].values
    else:
        raise AttributeError
    # lonlat for NIP
    indlon = np.array([np.where(lon0 == 99.875), np.where(lon0 == 150.125)]).flatten()
    lon0 = lon0[indlon[0]:indlon[1]+1]
    indlat = np.array([np.where(lat0 == -1.125), np.where(lat0 == 50.125)]).flatten()
    lat0 = lat0[indlat[0]:indlat[1]+1]
    data_real_4_evaluate = np.squeeze(ds[var]).values
    data_real_4_evaluate = data_real_4_evaluate[:,indlat[0]:indlat[1]+1,indlon[0]:indlon[1]+1]
    data_real_4_evaluate = data_real_4_evaluate.astype('float64')    
    return lon0,lat0,doys_predict,data_real_4_evaluate

def use_eof_to_calculate_pc(var,data_history,doys_history):
    # Calculate Principal Components (PC) using EOF analysis
    if var.lower() == 'sst':
        var_cli = scipy.io.loadmat('sstcli.mat')['sstcli']
        var_cli = np.transpose(var_cli,[2,1,0])
        nan_mask = np.isnan(var_cli)
        any_nan_mask = np.any(nan_mask, axis=0)
        mask = ~any_nan_mask
        ssta = data_history - var_cli[np.array(doys_history)-1]
        ssta1 = np.reshape(ssta,(len(doys_history),-1))
        ssta1 = ssta1[:,np.reshape(mask,(-1))]
        ssta1 = np.transpose(ssta1)
        ssta_eof_inv = scipy.io.loadmat('ssta_eof_1-1000.mat')['e1inv']
        ssta_eof_inv = ssta_eof_inv.astype(np.float64)
        var_pc = np.dot(ssta_eof_inv,ssta1)
        var_eof = scipy.io.loadmat('ssta_eof_1-1000.mat')['e1']
        var_res = (ssta1 - np.dot(var_eof,var_pc))[:,-1]
    elif var.lower() == 'adt':
        var_cli = scipy.io.loadmat('adtcli.mat')['adtcli']
        var_cli = np.transpose(var_cli,[2,1,0])
        nan_mask = np.isnan(var_cli)
        any_nan_mask = np.any(nan_mask, axis=0)
        mask = ~any_nan_mask
        adta = data_history - var_cli[np.array(doys_history)-1]
        adta1 = np.reshape(adta,(len(doys_history),-1))
        adta1 = adta1[:,np.reshape(mask,(-1))]
        adta1 = np.transpose(adta1)
        adta_eof_inv = scipy.io.loadmat('adta_eof_1-1000.mat')['e1inv']
        adta_eof_inv = adta_eof_inv.astype(np.float64)
        var_pc = np.dot(adta_eof_inv,adta1)
        var_eof = scipy.io.loadmat('adta_eof_1-1000.mat')['e1']
        var_res = (adta1 - np.dot(var_eof,var_pc))[:,-1]
    else:
        raise ValueError
        
    return var_pc,var_eof,var_res,mask,var_cli   
  
def use_pc_to_predict_data(var,data_history,days_history,days_predict,target_date,doys_history):
    var_pc,var_eof,var_res,mask,var_cli = use_eof_to_calculate_pc(var,data_history,doys_history)
    csvfile = f'{var}a-PCA1-15.csv'
    df = pd.read_csv(f'{csvfile}')
    pc_nums = np.size(df,axis=1) - 1
    df.head()
    pc_predicts_standard = np.full([pc_nums,days_predict],np.nan)
    pc_predicts = np.full([pc_nums,days_predict],np.nan)
    for k_pc in range(1,pc_nums+1):
        model = load_model(f'../model/finalmodel-{var}a-0/PC{k_pc}.h5')
        features = df[[f'PC{k_pc}']]
        features.index = df['Date']
        data0 = features.values
        pc_original = var_pc[k_pc-1,:].reshape(-1,1)
        data1 = np.concatenate((data0,pc_original),axis=0)
        pc_standard,scaler = data_standardization(data1)  
        pc_standard = pc_standard[-days_history:]
        x_input = pc_standard # default
        for k_step in range(1,days_predict+1):
            if k_step == 1:
                x_input = pc_standard
                pc_predict = model.predict(pc_standard.reshape(1,days_history,1))           
            else:
                x_input = np.delete(x_input,0, axis = 0)
                x_input = np.concatenate((x_input, pc_predict), axis = 0)
                pc_predict = model.predict(x_input.reshape(1,days_history,1))
            pc_predicts_standard[k_pc-1,k_step-1] = np.squeeze(pc_predict)
        pc_predicts[k_pc-1,:] = data_anti_standardization(pc_predicts_standard[k_pc-1,:],scaler)
        del data0  
    # data predicted by the 1-15th modes # 2D array
    data_predict = np.dot(var_eof,pc_predicts)
    # # data residual for the other higher modes
    var_res = np.tile(var_res[:, np.newaxis], (1, days_predict)) 
    # # total data predict
    data_predict = data_predict + var_res 
    # ## fill land nan value, data_ano_predict will be a 3D array
    data_ano_predict = np.full((data_predict.shape[1], mask.size), np.nan)
    data_ano_predict[:, mask.flatten()] = data_predict.T
    data_ano_predict = data_ano_predict.reshape(data_predict.shape[1], mask.shape[0], mask.shape[1])
    doys_predict,_ = get_doys_predict(target_date,days_predict)
    # # here the adt_predict is a 3D array
    data_predict = data_ano_predict + var_cli[doys_predict,:,:]
    return data_predict,data_ano_predict,var_cli

def plot_the_kth_Total_predict_and_true(long,latg,var,k_step,data_predict,data_real_4_evaluate):
    data1 = data_predict[k_step-1,:,:]
    data2 = data_real_4_evaluate[k_step-1,:,:]
    Xlon, Ylat = np.meshgrid(long, latg)
    # 创建一个绘图窗口
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))    
    if var.lower() == 'sst':
        # 设置色阶和色图
        levels = np.arange(-2, 45)
        cmap = plt.cm.jet
        # 绘制第一个数据集的等值线图
        clim = (10, 35)
        contour1 = axes[0].pcolormesh(Xlon, Ylat, data1, cmap=cmap, shading='nearest')       
        axes[0].contour(Xlon, Ylat, data1, levels = levels,colors='black', linewidths=0.5)
        axes[0].set_title(f'SST - Predict - Day{k_step}')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        cbar1 = fig.colorbar(contour1, ax=axes[0], orientation='vertical', label='Temperature (°C)',extend='both')
        cbar1.mappable.set_clim(vmin = clim[0],vmax = clim[1])
        cbar1.set_ticks(np.linspace(clim[0], clim[1], 11))  # 设置颜色条刻度
        cbar1.set_ticklabels([f'{i:.1f}°C' for i in np.linspace(clim[0], clim[1], 11)])  # 设置颜色条刻度标签
        # 绘制第二个数据集的等值线图
        contour2 = axes[1].pcolormesh(Xlon, Ylat, data2, cmap=cmap, shading='nearest')
        axes[1].contour(Xlon, Ylat, data2, levels = levels,colors='black', linewidths=0.5)
        axes[1].set_title(f'SST - True - Day{k_step}')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        cbar2 = fig.colorbar(contour2, ax=axes[1], orientation='vertical', label='Temperature (°C)')
        cbar2.mappable.set_clim(vmin = clim[0],vmax = clim[1])
        cbar2.set_ticks(np.linspace(clim[0], clim[1], 11))  # 设置颜色条刻度
        cbar2.set_ticklabels([f'{i:.1f}°C' for i in np.linspace(clim[0], clim[1], 11)])  # 设置颜色条刻度标签
    elif var.lower() == 'adt':
        # 设置色阶和色图
        levels = np.arange(-2, 2,0.2)
        cmap = plt.cm.jet
        # 绘制第一个数据集的等值线图
        clim = (-2.0, 2.0)
        contour1 = axes[0].pcolormesh(Xlon, Ylat, data1, cmap=cmap, shading='nearest')   
        axes[0].contour(Xlon, Ylat, data1, levels = levels,colors='black', linewidths=0.5)
        axes[0].set_title(f'ADT - Predict - Day{k_step}')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        cbar1 = fig.colorbar(contour1, ax=axes[0], orientation='vertical', label='ADT(m)')
        cbar1.mappable.set_clim(vmin = clim[0],vmax = clim[1])
        cbar1.set_ticks(np.linspace(clim[0], clim[1], 21))  # 设置颜色条刻度
        cbar1.set_ticklabels([f'{i:.1f}m' for i in np.linspace(clim[0], clim[1], 21)])  # 设置颜色条刻度标签
        # 绘制第二个数据集的等值线图
        contour2 = axes[1].pcolormesh(Xlon, Ylat, data2, cmap=cmap, shading='nearest') 
        axes[1].contour(Xlon, Ylat, data1, levels = levels,colors='black', linewidths=0.5)
        axes[1].set_title(f'ADT - True - Day{k_step}')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        cbar2 = fig.colorbar(contour2, ax=axes[1], orientation='vertical', label='ADT(m)')
        cbar2.mappable.set_clim(vmin = clim[0],vmax = clim[1])
        cbar2.set_ticks(np.linspace(clim[0], clim[1], 21))  # 设置颜色条刻度
        cbar2.set_ticklabels([f'{i:.1f}°m' for i in np.linspace(clim[0], clim[1], 21)])  # 设置颜色条刻度标签
    else:
        raise ValueError
   
def plot_the_kth_Anomaly_predict_and_true(long,latg,var,k_step,data_ano_predict,data_ano_real_4_evaluate):
    data1 = data_ano_predict[k_step-1,:,:]
    data2 = data_ano_real_4_evaluate[k_step-1,:,:]
    Xlon, Ylat = np.meshgrid(long, latg)
    # 创建一个绘图窗口
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    if var.lower() == 'sst':    
        levels = np.linspace(-5, 5, 21)
        ll = np.hstack(([-4,-3,-2], np.arange(-1.5,1.5,0.5), [2,3,4]))
        cmap = plt.cm.bwr       
        # 绘制第一个数据集的等值线图
        clim = (-5, 5)
        contourf1 = axes[0].pcolormesh(Xlon, Ylat, data1, cmap=cmap, shading='nearest')   
        contour1 = axes[0].contour(Xlon, Ylat, data1, levels = ll,colors='black', linewidths=0.5)
        axes[0].clabel(contour1, inline=True, fontsize=8, fmt='%2.1f')
        axes[0].set_title(f'SSTA - Predict - Day{k_step}')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        cbar1 = fig.colorbar(contourf1, ax=axes[0], orientation='vertical', label='Temperature (°C)')
        cbar1.mappable.set_clim(vmin = clim[0],vmax = clim[1])
        cbar1.set_ticks(np.linspace(clim[0], clim[1], 11))  # 设置颜色条刻度
        cbar1.set_ticklabels([f'{i:.2f}°C' for i in np.linspace(clim[0], clim[1], 11)])  # 设置颜色条刻度标签
        # 绘制第二个数据集的等值线图
        contourf2 = axes[1].pcolormesh(Xlon, Ylat, data2, cmap=cmap, shading='nearest')   
        contour2 = axes[1].contour(Xlon, Ylat, data2, levels=ll, colors='black', linewidths=0.5)
        axes[1].clabel(contour2, inline=True, fontsize=8, fmt='%2.1f')
        axes[1].set_title(f'SSTA - True - Day{k_step}')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        cbar2 = fig.colorbar(contourf2, ax=axes[1], orientation='vertical', label='Temperature (°C)')
        cbar2.mappable.set_clim(vmin = clim[0],vmax = clim[1])
        cbar2.set_ticks(np.linspace(clim[0], clim[1], 11))  # 设置颜色条刻度
        cbar2.set_ticklabels([f'{i:.2f}°C' for i in np.linspace(clim[0], clim[1], 11)])  # 设置颜色条刻度标签
    elif var.lower() == 'adt': 
        levels = np.linspace(-3, 3, 41)
        cmap = plt.cm.jet        
        # 绘制第一个数据集的等值线图
        clim = (-1.0, 1.0)
        contour1 = axes[0].pcolormesh(Xlon, Ylat, data1, cmap=cmap, shading='nearest')   
        axes[0].set_title(f'ADTA - Predict - Day{k_step}')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        cbar1 = fig.colorbar(contour1, ax=axes[0], orientation='vertical', label='ADTA(m)')
        cbar1.mappable.set_clim(vmin = clim[0],vmax = clim[1])
        cbar1.set_ticks(np.linspace(clim[0], clim[1], 21))  # 设置颜色条刻度
        cbar1.set_ticklabels([f'{i:.1f}m' for i in np.linspace(clim[0], clim[1], 21)])  # 设置颜色条刻度标签
        # 绘制第二个数据集的等值线图
        contour2 = axes[1].pcolormesh(Xlon, Ylat, data2, cmap=cmap, shading='nearest')   
        axes[1].set_title(f'ADTA - True - Day{k_step}')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        cbar2 = fig.colorbar(contour2, ax=axes[1], orientation='vertical', label='ADTA(m)')
        cbar2.mappable.set_clim(vmin = clim[0],vmax = clim[1])
        cbar2.set_ticks(np.linspace(clim[0], clim[1], 21))  # 设置颜色条刻度
        cbar2.set_ticklabels([f'{i:.1f}°m' for i in np.linspace(clim[0], clim[1], 21)])  # 设置颜色条刻度标签
    else:
        raise ValueError
    # 调整布局并显示图像
    plt.tight_layout()
    plt.show() 
    
def writeNC(data_out_file,var, long, latg, date_predict, data_predict, data_ano_predict): 
    
    f_w = nc.Dataset(data_out_file,'w',format = 'NETCDF4')
    f_w.createDimension('time',len(date_predict))
    f_w.createDimension('lat',len(latg))
    f_w.createDimension('lon',len(long))
    
    f_w.createVariable('time',np.int32,  ('time'))
    f_w.createVariable('lat', np.float32,('lat'))
    f_w.createVariable('lon', np.float32,('lon')) 
    if var.lower() == 'sst':  
        f_w.createVariable('sst', np.float64,('time','lat','lon'))   
        f_w.createVariable('ssta', np.float64,('time','lat','lon')) 
    elif var.lower() == 'adt':
        f_w.createVariable('adt', np.float64,('time','lat','lon'))   
        f_w.createVariable('adta', np.float64,('time','lat','lon')) 
    else:
        raise AttributeError
        
    f_w.variables['time'][:] = date_predict
    f_w.variables['lat'][:]  = latg
    f_w.variables['lon'][:]  = long
    if var.lower() == 'sst':  
        f_w.variables['sst'][:]  = data_predict
        f_w.variables['ssta'][:]  = data_ano_predict
    elif var.lower() == 'adt':
        f_w.variables['adt'][:]  = data_predict
        f_w.variables['adta'][:]  = data_ano_predict
    else:
        raise AttributeError
    f_w.close
                    
def funciton_4_main_predict(var, data_in_path, target_date, days_history, days_predict): 
    long,latg,doys_history,data_history = read_ncfiles_data_before_date(var, data_in_path, days_history, target_date)
    # prediction
    data_predict,data_ano_predict,var_cli = use_pc_to_predict_data(var,data_history,days_history,days_predict,target_date,doys_history)    
    return long,latg,data_predict,data_ano_predict,var_cli

def funciton_4_main_evaluate(var, data_in_path, target_date, days_predict, long, latg, data_predict, data_ano_predict, var_cli, k_step): 
    _,_,doys_predict,data_real_4_evaluate = read_ncfiles_data_after_date(var, data_in_path, days_predict, target_date)
    data_ano_real_4_evaluate = data_real_4_evaluate -  var_cli[doys_predict,:,:]
    statistics_4_data = calculate_statistics_4_evaluate(days_predict,data_real_4_evaluate,data_predict)
    statistics_4_data_ano = calculate_statistics_4_evaluate(days_predict,data_ano_real_4_evaluate,data_ano_predict)
    ## visualization
    plot_the_kth_Total_predict_and_true(long,latg,var,k_step,data_predict,data_real_4_evaluate)
    plot_the_kth_Anomaly_predict_and_true(long,latg,var,k_step,data_ano_predict,data_ano_real_4_evaluate)
    return statistics_4_data,statistics_4_data_ano
    
# [In]
# in_path = "U:/Observations/SST/NOAA_OISST/AVHRRv02r01/"
if __name__=='__main__':
    ############################################################### define something###########################################
    date_begin = dt.date(2022,9,21)
    date_end   = dt.date(2022,9,22)
    # date_end   = dt.date(2023,5,16)
    date_num = (date_end - date_begin).days
    data_in_path_SST = "G:/SST/NOAA_OISST/AVHRRv02r01/"
    data_in_path_SSH = "G:/SSH-vDT2021/CMEMS_L4_REP_allsat/"
    days_history = 30
    days_predict = 15
    #############################################################################################################################
    indicators_sst  = np.ones([days_predict,4,date_num],dtype = np.float64)
    indicators_ssta = np.ones([days_predict,4,date_num],dtype = np.float64)
    indicators_adt  = np.ones([days_predict,4,date_num],dtype = np.float64)
    indicators_adta = np.ones([days_predict,4,date_num],dtype = np.float64)
    for iday in range(0,date_num):
        # target_date = datetime(2021, 11, 21)
        target_date = date_begin + timedelta(days = iday)
        print(f'Now Prediction Date: {target_date.strftime("%Y%m%d")}')
        time_start = tim.time()    
        _,date_predict = get_doys_predict(target_date, days_predict)        
        ######################################################## SST #################################################
        var = 'sst'
        long,latg,sst_predict,ssta_predict,var_cli = funciton_4_main_predict(var, data_in_path_SST, target_date, days_history, days_predict)
        data_out_path = f'../output/{var}/{target_date.strftime("%Y")}'
        if not os.path.exists(data_out_path):
           os.makedirs(data_out_path)
        data_out_file = f'{data_out_path}\Prediction-{var}-{target_date.strftime("%Y%m%d")}.nc'
        writeNC(data_out_file, var, long, latg, date_predict, sst_predict, ssta_predict)
        k_step = 5
        statistics_4_sst,statistics_4_ssta = funciton_4_main_evaluate(var, data_in_path_SST, target_date, days_predict, long, latg,
                                                                           sst_predict, ssta_predict, var_cli, k_step)
        indicators_sst[...,iday]  = statistics_4_sst
        indicators_ssta[...,iday] = statistics_4_ssta
        print('SST Prediction finished')
        ######################################################## ADT #################################################
        var = 'adt'
        long,latg,adt_predict,adta_predict,var_cli = funciton_4_main_predict(var, data_in_path_SSH, target_date, days_history, days_predict)        
        data_out_path = f'../output/{var}/{target_date.strftime("%Y")}'
        if not os.path.exists(data_out_path):
           os.makedirs(data_out_path)
        data_out_file = f'{data_out_path}/Prediction-{var}-{target_date.strftime("%Y%m%d")}.nc'
        writeNC(data_out_file, var, long, latg, date_predict, adt_predict, adta_predict)
        k_step = 5
        statistics_4_adt,statistics_4_adta = funciton_4_main_evaluate(var, data_in_path_SSH, target_date, days_predict, long, latg,
                                                                           adt_predict, adta_predict, var_cli, k_step)
        indicators_adt[...,iday]  = statistics_4_adt
        indicators_adta[...,iday] = statistics_4_adta
        
        time_end = tim.time()
        minutes, seconds = divmod(time_end - time_start, 60)
        print(f"Prediction time: {int(minutes)}:{int(seconds)}")
    # savemat('../output/evaluated_indicators.mat',{'indicators_sst': indicators_sst, 'indicators_ssta': indicators_ssta,
    #                                               'indicators_adt': indicators_adt, 'indicators_adta': indicators_adta})
    
    #### old version ####
    # # read history data
    # long,latg,doys_history,data_history = read_ncfiles_data_before_date(var, data_in_path_SST, days_history, target_date)
    # # prediction
    # data_predict,data_ano_predict,var_cli = use_pc_to_predict_data(var,data_history,days_history,days_predict,target_date)
    # # evaluate, if possible
    # _,_,doys_predict,data_real_4_evaluate = read_ncfiles_data_after_date(var, data_in_path_SST, days_predict, target_date)
    # data_ano_real_4_evaluate = data_real_4_evaluate -  var_cli[doys_predict,:,:]
    # statistics_4_sst = calculate_statistics_4_evaluate(days_predict,data_real_4_evaluate,data_predict)
    # statistics_4_ssta = calculate_statistics_4_evaluate(days_predict,data_ano_real_4_evaluate,data_ano_predict)
    # ## visualization
    # k_step = 2
    # plot_the_kth_Total_predict_and_true(long,latg,var,k_step,data_predict,data_real_4_evaluate)
    # plot_the_kth_Anomaly_predict_and_true(long,latg,var,k_step,data_ano_predict,data_ano_real_4_evaluate)
    
    # ## predict ADT
    # var = 'adt'
    # # read history data
    # long,latg,doys_history,data_history = read_ncfiles_data_before_date(var, data_in_path_SSH, days_history, target_date)
    # # prediction
    # adt_predict,adt_ano_predict,var_cli = use_pc_to_predict_data(var,data_history,days_history,days_predict,target_date)
    # # evaluate, if possible
    # _,_,doys_predict,data_real_4_evaluate = read_ncfiles_data_after_date(var, data_in_path_SSH, days_predict, target_date)
    # data_ano_real_4_evaluate = data_real_4_evaluate -  var_cli[doys_predict,:,:]
    # statistics_4_adt = calculate_statistics_4_evaluate(days_predict,data_real_4_evaluate,adt_predict)
    # statistics_4_adta = calculate_statistics_4_evaluate(days_predict,data_ano_real_4_evaluate,adt_ano_predict)
    # ## visualization
    # k_step = 14
    # plot_the_kth_Total_predict_and_true(long,latg,var,k_step,adt_predict,data_real_4_evaluate)
    # plot_the_kth_Anomaly_predict_and_true(long,latg,var,k_step,adt_ano_predict,data_ano_real_4_evaluate)
    # time_end = tim.time()
    # minutes, seconds = divmod(time_end - time_start, 60)
    # print(f"Prediction time: {int(minutes)}:{int(seconds)}")
    # # write the prediction result
    # _,date_predict = get_doys_predict(target_date, days_predict)
    # data_out_file = f'{data_out_path}/Prediction-{target_date.strftime("%Y%m%d")}.nc'
    # writeNC(data_out_file, var, long, latg, date_predict, data_predict, data_ano_predict)