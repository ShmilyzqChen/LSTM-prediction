clear,clc
% read sst
ncfiles = dir('U:\Observations\SST\NOAA_OISST\AVHRRv02r01\*\*.nc');
kth = nan(2,1);
for k = 1:length(ncfiles)
    if strcmp(ncfiles(k).name,'oisst-avhrr-v02r01.19930101.nc')
        kth(1) = k;
    end
    if strcmp(ncfiles(k).name,'oisst-avhrr-v02r01.20221231.nc')
        kth(2) = k;
    end
end
ncfiles = ncfiles(kth(1):kth(2));
ncfile = 'U:\Observations\SST\NOAA_OISST\AVHRRv02r01\198212\oisst-avhrr-v02r01.19821208.nc';
[info,dims] = ncfile2mat(ncfile);
nipvarlim = [dims(1).Dimsfirvalue dims(1).Dimsendvalue 0 0 -20 70 40 180];
[lat,lon,sst] = ncfile2mat(ncfile,4:6,nipvarlim);
ln = length(ncfiles);
ssts = nan(size(sst,1),size(sst,2),ln);
for k = 1:length(ncfiles)
    ncfile = fullfile(ncfiles(k).folder,ncfiles(k).name);
    if mod(k,100)==0
        disp(ncfile)
    end
    ssts(:,:,k) = ncfile2mat(ncfile,{'sst'},nipvarlim);
end
% write sst
ncfile = 'NIP-SSTs2.nc';
lon0 = lon; lat0 = lat; 
daytimes = datenum(1993,1,1):datenum(2022,12,31);
mod_write_sst_ncfile(ncfile,lon0,lat0,daytimes,ssts); 
clear lon0 lat0 daytimes
ssts = ncread(ncfile,'ssts');
%%
clear,clc
% read ssh
ncfiles = dir('U:\Observations\SSH\SSH-vDT2021\CMEMS_L4_REP_allsat\*\*.nc');
kth = nan(2,1);
for k = 1:length(ncfiles)
    if strcmp(ncfiles(k).name,'dt_global_allsat_phy_l4_19930101_20210726.nc')
        kth(1) = k;
    end
    if strcmp(ncfiles(k).name,'dt_global_allsat_phy_l4_20221231_20231013.nc')
        kth(2) = k;
    end
end
ncfiles = ncfiles(kth(1):kth(2));
ncfile = 'U:\Observations\SSH\SSH-vDT2021\CMEMS_L4_REP_allsat\2022\dt_global_allsat_phy_l4_20221231_20231013.nc';
[info,dims] = ncfile2mat(ncfile);
nipvarlim = [dims(1).Dimsfirvalue dims(1).Dimsendvalue -20 70 40 179.875 0 1];
[lat,lon1,ssh1] = ncfile2mat(ncfile,[3 5 8],nipvarlim);
nipvarlim = [dims(1).Dimsfirvalue dims(1).Dimsendvalue -20 70 -179.875 -179.625 0 1];
[lon2,ssh2] = ncfile2mat(ncfile,[5 8],nipvarlim);
lon = [lon1;lon2(1:2)+360];
ssh = [ssh1;ssh2(1:2,:)];
ln = length(ncfiles);
adts = nan(size(ssh,1),size(ssh,2),ln);
for k = 1:length(ncfiles)
    ncfile = fullfile(ncfiles(k).folder,ncfiles(k).name);
    if mod(k,100)==0
        disp(ncfile)
    end
    nipvarlim = [dims(1).Dimsfirvalue dims(1).Dimsendvalue -20 70 40 179.875 0 1];
    adt1 = ncfile2mat(ncfile,{'adt'},nipvarlim);
    nipvarlim = [dims(1).Dimsfirvalue dims(1).Dimsendvalue -20 70 -179.875 -179.625 0 1];
    adt2 = ncfile2mat(ncfile,{'adt'},nipvarlim);
    adt = [adt1;adt2(1:2,:)];
    adts(:,:,k) = adt; 
    clear adt1 adt2 adt 
end
% write SSH
% ncfile = 'NIP-ADTs.nc';
% lon0 = lon; lat0 = lat; 
% daytimes = datenum(1993,1,1):datenum(2022,12,31);
% mod_write_ssh_ncfile(ncfile,lon0,lat0,daytimes,adts); 
% slas = nan(size(ssh,1),size(ssh,2),ln);
% for k = 1:length(ncfiles)
%     ncfile = fullfile(ncfiles(k).folder,ncfiles(k).name);
%     if mod(k,100)==0
%         disp(ncfile)
%     end
%     nipvarlim = [dims(1).Dimsfirvalue dims(1).Dimsendvalue -20 70 40 179.875 0 1];
%     sla1 = ncfile2mat(ncfile,{'sla'},nipvarlim);
%     nipvarlim = [dims(1).Dimsfirvalue dims(1).Dimsendvalue -20 70 -179.875 -179.625 0 1];
%     sla2 = ncfile2mat(ncfile,{'sla'},nipvarlim);
%     sla = [sla1;sla2(1:2,:)];
%     slas(:,:,k) = sla;
%     clear sla1 sla2 sla
% end
% ncfile = 'NIP-SLAs.nc';
% mod_write_ssh_ncfile(ncfile,lon0,lat0,daytimes,slas); 
% clear lon0 lat0 daytimes 