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
# import warnings
# warnings.filterwarnings("ignore")

def calculate_NSE(data1,data2):
    #function for calculating Nash-Sutcliffe Efficiency, NSE, equal to function r2score
    # The NSE is closer to 1, the better
    numerator = np.sum(np.power(data1-data2,2))
    denominator = np.sum(np.power(data1-np.mean(data1),2))
    nse = 1 - numerator/denominator
    return nse

def data_standardization(data0):
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data0)
    return data, scaler

def data_anti_standardization(data,scaler):
    # 数据反归一化
    data0 = scaler.inverse_transform(data.reshape(1,-1)) 
    return data0

def get_doys_predict(reference_date,days_predict):
    # get the day of the year of the predict date, for the total SST
    doys_predict = []
    for day in range(1,days_predict+1):
        traget_day = reference_date + timedelta(days=day)
        doy = traget_day.timetuple().tm_yday - 1
        doys_predict.append(doy)
    return doys_predict

def calculate_statistics_4_evaluate(days_predict,y_true,y_predict):
    # List four statistic indicators for data evaluate
    statistics_4_out = np.full([days_predict,4],np.nan)
    # 0-mae 1-rmse 2-r2score or named Nash-Sutcliffe Efficiency  3-P-correlation
    for k_step in range(0,days_predict):
        data1 = y_true[k_step,:,:].reshape(-1)                 # real
        data2 = y_predict[k_step,:,:].reshape(-1)                       # predict
        valid_indices = ~np.isnan(data1) & ~np.isnan(data2)
        data1 = data1[valid_indices]
        data2 = data2[valid_indices]
        # MAE
        statistics_4_out[k_step,0] = mean_absolute_error(data1, data2)     
        # RMSE
        statistics_4_out[k_step,1] = np.sqrt(mean_squared_error(data1, data2)) 
        # NSE
        statistics_4_out[k_step,2] = r2_score(data1, data2)      
        # R        
        statistics_4_out[k_step,3] = np.corrcoef(data1, data2)[0,1]         
    return statistics_4_out     
 
def get_SSTfiles_list_before_date(data_in_path_SST, days_history, reference_date):
    SST_head_name = 'oisst-avhrr-v02r01.'
    SST_files = []
    doys_history = []
    for day in range(days_history,-1,-1):
        traget_day = reference_date - timedelta(days=day)
        mm = traget_day.strftime("%Y%m")
        ymd = traget_day.strftime("%Y%m%d")
        SST_file = f'{data_in_path_SST}{mm}/{SST_head_name}{ymd}.nc'
        doy = traget_day.timetuple().tm_yday
        if os.path.exists(SST_file):
            SST_files.append(SST_file)
            doys_history.append(doy)
        else:
            raise ValueError(f'{SST_file} does not exist, Please add it in the target directory')
    return SST_files,doys_history

def get_SSHfiles_list_before_date(data_in_path_SSH, days_history, reference_date):
    SSH_head_name = 'dt_global_allsat_phy_l4_'
    SSH_files = []
    doys_history = []
    for day in range(days_history,-1,-1):
        traget_day = reference_date - timedelta(days=day)
        # mm = traget_day.strftime("%Y%m")
        yy = traget_day.strftime("%Y")
        ymd = traget_day.strftime("%Y%m%d")
        doy = traget_day.timetuple().tm_yday
        path1 = f'{data_in_path_SSH}/{yy}'
        SSH_file = f'{SSH_head_name}{ymd}_*.nc'
        for f_name in os.listdir(path1):
            if fnmatch.fnmatch(f_name, SSH_file):               
                SSH_file = f'{path1}/{f_name}'
                SSH_files.append(SSH_file)
                doys_history.append(doy)
            else:
                raise ValueError(f'{SSH_file} does not exist, Please add it in the target directory')
    return SSH_files,doys_history

def read_ncfiles_data_before_date(nc_files,var):
    ds = xr.open_mfdataset(nc_files)
    if var == 'sst': 
        lon0 = ds['lon'].values
        lat0 = ds['lat'].values
    else:
        lon0 = ds['longitude'].values
        lat0 = ds['latitude'].values    
    # lonlat for NIP
    indlon = np.array([np.where(lon0 == 99.875), np.where(lon0 == 150.125)]).flatten()
    lon0 = lon0[indlon[0]:indlon[1]+1]
    indlat = np.array([np.where(lat0 == -1.125), np.where(lat0 == 50.125)]).flatten()
    lat0 = lat0[indlat[0]:indlat[1]+1]
    ncfiles_history = np.squeeze(ds[var]).values
    ncfiles_history = ncfiles_history[:,indlat[0]:indlat[1]+1,indlon[0]:indlon[1]+1]
    return lon0,lat0,ncfiles_history

def read_SST_data_after_date(data_in_path_SST, days_predict, reference_date):
    SST_head_name = 'oisst-avhrr-v02r01.'
    SST_files = []
    for day in range(1,days_predict+1):
        traget_day = reference_date + timedelta(days=day)
        mm = traget_day.strftime("%Y%m")
        ymd = traget_day.strftime("%Y%m%d")
        SST_file = f'{data_in_path_SST}{mm}/{SST_head_name}{ymd}.nc'        
        if os.path.exists(SST_file):
            SST_files.append(SST_file)            
        else:
            raise ValueError(f'{SST_file} does not exist, Please add it in the target directory')
    ds = xr.open_mfdataset(SST_files)    
    lon0 = ds['lon'].values
    lat0 = ds['lat'].values
    # lonlat for NIP
    indlon = np.array([np.where(lon0 == 99.875), np.where(lon0 == 150.125)]).flatten()
    lon0 = lon0[indlon[0]:indlon[1]+1]
    indlat = np.array([np.where(lat0 == -1.125), np.where(lat0 == 50.125)]).flatten()
    lat0 = lat0[indlat[0]:indlat[1]+1]
    ssts_real_4_evaluate = np.squeeze(ds[var]).values
    ssts_real_4_evaluate = ssts_real_4_evaluate[:,indlat[0]:indlat[1]+1,indlon[0]:indlon[1]+1]
    return ssts_real_4_evaluate

def read_SSH_data_after_date(data_in_path_SSH, days_predict, reference_date,var):
    SSH_head_name = 'dt_global_allsat_phy_l4_'
    SSH_files = []
    for day in range(1,days_predict+1):
        traget_day = reference_date + timedelta(days=day)
        yy = traget_day.strftime("%Y")
        ymd = traget_day.strftime("%Y%m%d")
        path1 = f'{data_in_path_SSH}/{yy}'
        SSH_file = f'{SSH_head_name}{ymd}_*.nc'
        for f_name in os.listdir(path1):
            if fnmatch.fnmatch(f_name, SSH_file):               
                SSH_file = f'{path1}/{f_name}'
                SSH_files.append(SSH_file)           
    ds = xr.open_mfdataset(SSH_files)    
    lon0 = ds['longitude'].values
    lat0 = ds['latitude'].values
    # lonlat for NIP
    indlon = np.array([np.where(lon0 == 99.875), np.where(lon0 == 150.125)]).flatten()
    lon0 = lon0[indlon[0]:indlon[1]+1]
    indlat = np.array([np.where(lat0 == -1.125), np.where(lat0 == 50.125)]).flatten()
    lat0 = lat0[indlat[0]:indlat[1]+1]
    adts_real_4_evaluate = np.squeeze(ds[var]).values
    adts_real_4_evaluate = adts_real_4_evaluate[:,indlat[0]:indlat[1]+1,indlon[0]:indlon[1]+1]
    return adts_real_4_evaluate

def use_eof_to_calculate_sst_pc(ssts_history):
    sstcli = scipy.io.loadmat('sstcli.mat')['sstcli']
    sstcli = np.transpose(sstcli,[2,1,0])
    nan_mask = np.isnan(sstcli)
    any_nan_mask = np.any(nan_mask, axis=0)
    mask = ~any_nan_mask
    ssta = ssts_history - sstcli[np.array(doys_history)-1]
    ssta1 = np.reshape(ssta,(len(doys_history),-1))
    ssta1 = ssta1[:,np.reshape(mask,(-1))]
    ssta1 = np.transpose(ssta1)
    ssta_eof_inv = scipy.io.loadmat('ssta_eof_1-1000.mat')['e1inv']
    ssta_eof_inv = ssta_eof_inv.astype(np.float64)
    sst_pc = np.dot(ssta_eof_inv,ssta1)
    ssta_eof = scipy.io.loadmat('ssta_eof_1-1000.mat')['e1']
    sst_res = (ssta1 - np.dot(ssta_eof,sst_pc))[:,-1]
    return sst_pc,ssta_eof,sst_res,mask,sstcli

def use_eof_to_calculate_adt_pc(adts_history):
    adtcli = scipy.io.loadmat('adtcli.mat')['adtcli']
    adtcli = np.transpose(adtcli,[2,1,0])
    nan_mask = np.isnan(adtcli)
    any_nan_mask = np.any(nan_mask, axis=0)
    mask = ~any_nan_mask
    adta = adts_history - adtcli[np.array(doys_history)-1]
    adta1 = np.reshape(adta,(len(doys_history),-1))
    adta1 = adta1[:,np.reshape(mask,(-1))]
    adta1 = np.transpose(adta1)
    adta_eof_inv = scipy.io.loadmat('adta_eof_1-1000.mat')['e1inv']
    adta_eof_inv = adta_eof_inv.astype(np.float64)
    adt_pc = np.dot(adta_eof_inv,adta1)
    adta_eof = scipy.io.loadmat('adta_eof_1-1000.mat')['e1']
    adt_res = (adta1 - np.dot(adta_eof,adt_pc))[:,-1]
    return adt_pc,adta_eof,adt_res,mask,adtcli

def plot_the_kth_SST_predict_and_true(long,latg,k_step,sst_predict,ssts_real_4_evaluate):
    Xlon, Ylat = np.meshgrid(long, latg)
    # 创建一个绘图窗口
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    # 设置色阶和色图
    levels = np.linspace(10, 35, 26)
    cmap = plt.cm.jet
    data1 = sst_predict[k_step-1,:,:]
    data2 = ssts_real_4_evaluate[k_step-1,:,:]
    # 绘制第一个数据集的等值线图
    clim = (10, 35)
    contour1 = axes[0].contourf(Xlon, Ylat, data1, levels = levels, cmap=cmap, vmin = 0, vmax = 40)
    contour1.set_clim(*clim)
    axes[0].set_title(f'SST - Predict - Day{k_step}')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    cbar1 = fig.colorbar(contour1, ax=axes[0], orientation='vertical', label='Temperature (°C)')
    cbar1.set_ticks(np.linspace(clim[0], clim[1], 11))  # 设置颜色条刻度
    cbar1.set_ticklabels([f'{i:.1f}°C' for i in np.linspace(clim[0], clim[1], 11)])  # 设置颜色条刻度标签
    # 绘制第二个数据集的等值线图
    contour2 = axes[1].contourf(Xlon, Ylat, data2, levels = levels, cmap=cmap, vmin = 0, vmax = 40)
    contour2.set_clim(*clim)
    axes[1].set_title(f'SST - True - Day{k_step}')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    cbar2 = fig.colorbar(contour2, ax=axes[1], orientation='vertical', label='Temperature (°C)')
    cbar2.set_ticks(np.linspace(clim[0], clim[1], 11))  # 设置颜色条刻度
    cbar2.set_ticklabels([f'{i:.1f}°C' for i in np.linspace(clim[0], clim[1], 11)])  # 设置颜色条刻度标签
    # 调整布局并显示图像
    plt.tight_layout()
    plt.show()
    
    
def plot_the_kth_SSTA_predict_and_true(long,latg,k_step,ssta_predict,ssta_real_4_evaluate):
    Xlon, Ylat = np.meshgrid(long, latg)
    # 创建一个绘图窗口
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    # 设置色阶和色图
    levels = np.linspace(-5, 5, 21)
    ll = np.hstack(([-4,-3,-2], np.arange(-1.5,1.5,0.5), [2,3,4]))
    cmap = plt.cm.bwr
    data1 = ssta_predict[k_step-1,:,:]
    data2 = ssta_real_4_evaluate[k_step-1,:,:]
    # 绘制第一个数据集的等值线图
    clim = (-5, 5)
    contourf1 = axes[0].contourf(Xlon, Ylat, data1, levels = levels, cmap=cmap, vmin = -5, vmax = 5)
    contourf1.set_clim(*clim)
    contour1 = axes[0].contour(Xlon, Ylat, data1, levels = ll,colors='black', linewidths=0.5)
    axes[0].clabel(contour1, inline=True, fontsize=8, fmt='%2.1f')
    axes[0].set_title(f'SSTA - Predict - Day{k_step}')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    cbar1 = fig.colorbar(contourf1, ax=axes[0], orientation='vertical', label='Temperature (°C)')
    cbar1.set_ticks(np.linspace(clim[0], clim[1], 11))  # 设置颜色条刻度
    cbar1.set_ticklabels([f'{i:.2f}°C' for i in np.linspace(clim[0], clim[1], 11)])  # 设置颜色条刻度标签
    # 绘制第二个数据集的等值线图
    contourf2 = axes[1].contourf(Xlon, Ylat, data2, levels = levels, cmap=cmap, vmin = -5, vmax = 5)
    contourf2.set_clim(*clim)
    contour2 = axes[1].contour(Xlon, Ylat, data2, levels=ll, colors='black', linewidths=0.5)
    axes[1].clabel(contour2, inline=True, fontsize=8, fmt='%2.1f')
    axes[1].set_title(f'SSTA - True - Day{k_step}')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    cbar2 = fig.colorbar(contourf2, ax=axes[1], orientation='vertical', label='Temperature (°C)')
    cbar2.set_ticks(np.linspace(clim[0], clim[1], 11))  # 设置颜色条刻度
    cbar2.set_ticklabels([f'{i:.2f}°C' for i in np.linspace(clim[0], clim[1], 11)])  # 设置颜色条刻度标签
    # 调整布局并显示图像
    plt.tight_layout()
    plt.show()    

def plot_the_kth_SSH_predict_and_true(long,latg,k_step,adt_predict,adts_real_4_evaluate):
    Xlon, Ylat = np.meshgrid(long, latg)
    # 创建一个绘图窗口
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    # 设置色阶和色图
    levels = np.linspace(-3, 3, 21)
    cmap = plt.cm.jet
    data1 = adt_predict[k_step-1,:,:]
    data2 = adts_real_4_evaluate[k_step-1,:,:]
    # 绘制第一个数据集的等值线图
    clim = (-2.0, 2.0)
    contour1 = axes[0].contourf(Xlon, Ylat, data1, levels = levels, cmap=cmap, vmin = -2.0, vmax = 2.0)
    contour1.set_clim(*clim)
    axes[0].set_title(f'ADT - Predict - Day{k_step}')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    cbar1 = fig.colorbar(contour1, ax=axes[0], orientation='vertical', label='ADT(m)')
    cbar1.set_ticks(np.linspace(clim[0], clim[1], 21))  # 设置颜色条刻度
    cbar1.set_ticklabels([f'{i:.1f}m' for i in np.linspace(clim[0], clim[1], 21)])  # 设置颜色条刻度标签
    # 绘制第二个数据集的等值线图
    contour2 = axes[1].contourf(Xlon, Ylat, data2, levels = levels, cmap=cmap, vmin = -2.0, vmax = 2.0)
    contour2.set_clim(*clim)
    axes[1].set_title(f'ADT - True - Day{k_step}')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    cbar2 = fig.colorbar(contour2, ax=axes[1], orientation='vertical', label='ADT(m)')
    cbar2.set_ticks(np.linspace(clim[0], clim[1], 21))  # 设置颜色条刻度
    cbar2.set_ticklabels([f'{i:.1f}°m' for i in np.linspace(clim[0], clim[1], 21)])  # 设置颜色条刻度标签
    # 调整布局并显示图像
    plt.tight_layout()
    plt.show()
    
def plot_the_kth_SSHA_predict_and_true(long,latg,k_step,adta_predict,adta_real_4_evaluate):
    Xlon, Ylat = np.meshgrid(long, latg)
    # 创建一个绘图窗口
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    # 设置色阶和色图
    levels = np.linspace(-3, 3, 41)
    cmap = plt.cm.jet
    data1 = adta_predict[k_step-1,:,:]
    data2 = adta_real_4_evaluate[k_step-1,:,:]
    # 绘制第一个数据集的等值线图
    clim = (-1.0, 1.0)
    contour1 = axes[0].contourf(Xlon, Ylat, data1, levels = levels, cmap=cmap, vmin = -1.0, vmax = 1.0)
    contour1.set_clim(*clim)
    axes[0].set_title(f'ADTA - Predict - Day{k_step}')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    cbar1 = fig.colorbar(contour1, ax=axes[0], orientation='vertical', label='ADTA(m)')
    cbar1.set_ticks(np.linspace(clim[0], clim[1], 21))  # 设置颜色条刻度
    cbar1.set_ticklabels([f'{i:.1f}m' for i in np.linspace(clim[0], clim[1], 21)])  # 设置颜色条刻度标签
    # 绘制第二个数据集的等值线图
    contour2 = axes[1].contourf(Xlon, Ylat, data2, levels = levels, cmap=cmap, vmin = -1.0, vmax = 1.0)
    contour2.set_clim(*clim)
    axes[1].set_title(f'ADTA - True - Day{k_step}')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    cbar2 = fig.colorbar(contour2, ax=axes[1], orientation='vertical', label='ADTA(m)')
    cbar2.set_ticks(np.linspace(clim[0], clim[1], 21))  # 设置颜色条刻度
    cbar2.set_ticklabels([f'{i:.1f}°m' for i in np.linspace(clim[0], clim[1], 21)])  # 设置颜色条刻度标签
    # 调整布局并显示图像
    plt.tight_layout()
    plt.show()
    
# [In]
# in_path = "U:/Observations/SST/NOAA_OISST/AVHRRv02r01/"
data_in_path_SST = "G:/SST/NOAA_OISST/AVHRRv02r01/"
data_in_path_SSH = "G:/SSH-vDT2021/CMEMS_L4_REP_allsat/"
reference_date = datetime(2021, 11, 21)
days_history = 30
days_predict = 15
# var = 'sst'
# SST_files,doys_history = get_SSTfiles_list_before_date(data_in_path_SST, days_history, reference_date)
# long,latg,ssts_history = read_ncfiles_data_before_date(SST_files,var)
# sst_pc,ssta_eof,sst_res,mask,sstcli = use_eof_to_calculate_sst_pc(ssts_history)
# csvfile = 'ssta-PCA1-15.csv'
# df = pd.read_csv(f'{csvfile}')
# df.head()
# pc_predicts_standard = np.full([days_predict,days_predict],np.nan)
# pc_predicts = np.full([days_predict,days_predict],np.nan)
# for k_pc in range(1,days_predict+1):
#     model = load_model(f'./finalmodel-ssta/PC{k_pc}.h5')
#     PCstr = f'PC{k_pc}'
#     features_considered = [PCstr]
#     features = df[features_considered]
#     features.index = df['Date']
#     data0 = features.values
#     pc_original = sst_pc[k_pc-1,:].reshape(-1,1)
#     data1 = np.concatenate((data0,pc_original),axis=0)
#     pc_standard,scaler = data_standardization(data1)  
#     pc_standard = pc_standard[-days_history:]
#     x_input = pc_standard # default
#     for k_step in range(1,days_predict+1):
#         if k_step == 1:
#             x_input = pc_standard
#             pc_predict = model.predict(pc_standard.reshape(1,days_history,1))           
#         else:
#             x_input = np.delete(x_input,0, axis = 0)
#             x_input = np.concatenate((x_input, pc_predict), axis = 0)
#             pc_predict = model.predict(x_input.reshape(1,days_history,1))
#         pc_predicts_standard[k_pc-1,k_step-1] = np.squeeze(pc_predict)
#     pc_predicts[k_pc-1,:] = data_anti_standardization(pc_predicts_standard[k_pc-1,:],scaler)
#     del data0  
#     # plt.plot(range(days_history+1),sst_pc[k_pc-1,:],'blue', label='observed')
#     # plt.plot(range(days_history,days_history+days_predict),pc_predicts[k_pc-1,:],color='red',label='predict')
#     # plt.legend()

# # sst predicted by the 1-15th modes # 2D array
# sst_predict = np.dot(ssta_eof,pc_predicts)
# # sst residual for the other modes
# sst_res = np.tile(sst_res[:, np.newaxis], (1, days_predict)) 
# # total sst predict
# sst_predict = sst_predict + sst_res 
# ## fill land nan value, ssta_predict will be 3D array
# ssta_predict = np.full((sst_predict.shape[1], mask.size), np.nan)
# ssta_predict[:, mask.flatten()] = sst_predict.T
# ssta_predict = ssta_predict.reshape(sst_predict.shape[1], mask.shape[0], mask.shape[1])
# doys_predict = get_doys_predict(reference_date,days_predict)
# # here the sst_predict is a 3D array
# sst_predict = ssta_predict + sstcli[doys_predict,:,:] 

# ## predict performance check
# ssts_real_4_evaluate = read_SST_data_after_date(data_in_path_SST, days_predict, reference_date)
# ssts_real_4_evaluate = ssts_real_4_evaluate.astype('float64')
# ssta_real_4_evaluate = ssts_real_4_evaluate - sstcli[doys_predict,:,:]
# # total SST field evaluate
# statistics_4_sst = calculate_statistics_4_evaluate(days_predict,ssts_real_4_evaluate,sst_predict)
# # SST anomaly field evaluate
# statistics_4_ssta = calculate_statistics_4_evaluate(days_predict,ssta_real_4_evaluate,ssta_predict)

# k_step = 2
# plot_the_kth_SST_predict_and_true(long,latg,k_step,sst_predict,ssts_real_4_evaluate)
# plot_the_kth_SSTA_predict_and_true(long,latg,k_step,ssta_predict,ssta_real_4_evaluate)

## predict ADT
var = 'adt'
SSH_files,doys_history = get_SSHfiles_list_before_date(data_in_path_SSH, days_history, reference_date)
long,latg,adts_history = read_ncfiles_data_before_date(SSH_files,var)
adt_pc,adta_eof,adt_res,mask,adtcli = use_eof_to_calculate_adt_pc(adts_history)
csvfile = 'adta-PCA1-15.csv'
df = pd.read_csv(f'{csvfile}')
df.head()
pc_predicts_standard = np.full([days_predict,days_predict],np.nan)
pc_predicts = np.full([days_predict,days_predict],np.nan)
for k_pc in range(1,days_predict+1):
    model = load_model(f'./finalmodel-adta/PC{k_pc}.h5')
    PCstr = f'PC{k_pc}'
    features_considered = [PCstr]
    features = df[features_considered]
    features.index = df['Date']
    data0 = features.values
    pc_original = adt_pc[k_pc-1,:].reshape(-1,1)
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
# adt predicted by the 1-15th modes # 2D array
adt_predict = np.dot(adta_eof,pc_predicts)
# # adt residual for the other modes
adt_res = np.tile(adt_res[:, np.newaxis], (1, days_predict)) 
# # total adt predict
adt_predict = adt_predict + adt_res 
# ## fill land nan value, adta_predict will be 3D array
adta_predict = np.full((adt_predict.shape[1], mask.size), np.nan)
adta_predict[:, mask.flatten()] = adt_predict.T
adta_predict = adta_predict.reshape(adt_predict.shape[1], mask.shape[0], mask.shape[1])
doys_predict = get_doys_predict(reference_date,days_predict)
# # here the adt_predict is a 3D array
adt_predict = adta_predict + adtcli[doys_predict,:,:] 

# ## predict performance check
adts_real_4_evaluate = read_SSH_data_after_date(data_in_path_SSH, days_predict, reference_date,var)
adts_real_4_evaluate = adts_real_4_evaluate.astype('float64')
adta_real_4_evaluate = adts_real_4_evaluate - adtcli[doys_predict,:,:]
# # total ADT field evaluate
statistics_4_adt = calculate_statistics_4_evaluate(days_predict,adts_real_4_evaluate,adt_predict)
# # ADT anomaly field evaluate
statistics_4_adta = calculate_statistics_4_evaluate(days_predict,adta_real_4_evaluate,adta_predict)

k_step = 14
plot_the_kth_SSH_predict_and_true(long,latg,k_step,adt_predict,adts_real_4_evaluate)
plot_the_kth_SSHA_predict_and_true(long,latg,k_step,adta_predict,adta_real_4_evaluate)