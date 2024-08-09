clear,clc
ncfile = 'NIP-SSTs.nc';
[lon,lat,tt,ssts] = ncfile2mat(ncfile,1:4);
lon = lon(242:443);
lat = lat(78:283);
sstyear = ssts(242:443,78:283,:);
% clear ssts
% a = sstyear(:,:,10000);
%% sst
    [eof_maps0,pc0,V0,expvar0] = myeof(sstyear);
    EOF0 = fliplr(V0);
    PCA0 = flipud(pc0);
%     A0 = (EOF0(:,1:end)*PCA0(1:end,:))'; % Restore the original data field
%     mask = ~any(isnan(ssta),3);
%     Arestored = nan(size(A0,1),numel(mask));
%     Arestored(:,mask(:)) = A0;
%     Arestored = reshape(Arestored,size(A0,1),size(mask,1),size(mask,2));
%     Arestored = permute(Arestored,[2 3 1]);
%     k = 10000;
%     yval = Arestored(:,:,k);
%     y0yc = nanmean(sstyear,3);
%     ysum = y0yc +Arestored(:,:,k);
%     ydiff = sstyear(:,:,k) - ysum;
%     clear A1 V1 pc1
%% ssta = sst-sst_climatlolgy, ssta no detrend
    sstcli = climatology(sstyear,tt);
    ssta = nan(size(sstyear));
    ttnum = doy(tt);
    for k = 1:366
        ind = find(ttnum==k);
        ssta(:,:,ind) = sstyear(:,:,ind)-sstcli(:,:,k);
    end
    [eof_maps1,pc1,V1,expvar1] = myeof_nodetrend(ssta);
    EOF1 = fliplr(V1);
    PCA1 = flipud(pc1);
    EOF1 = single(EOF1);
    PCA1 = single(PCA1);
    A1 = (EOF1(:,1:10000)*PCA1(1:10000,:))'; % Restore the original data field
    mask = ~any(isnan(ssta),3);
    Arestored = nan(size(A1,1),numel(mask));
    Arestored(:,mask(:)) = A1;
    Arestored = reshape(Arestored,size(A1,1),size(mask,1),size(mask,2));
    Arestored = permute(Arestored,[2 3 1]);
    k = 500;
    yval = Arestored(:,:,k);
    y0yc = sstcli(:,:,doy(tt(k)));
    ysum = y0yc +Arestored(:,:,k);
    ydiff = sstyear(:,:,k) - ysum;
    clear A1 V1 pc1
%     save('ssta-eof.mat','EOF1');
%     save('ssta-pca.mat','PCA1');
tstr = datetime(datestr(tt,'yyyymmdd'),'InputFormat','yyyyMMdd','Format','preserveinput');

%列名称hao
col={'Date','PC1', 'PC2', 'PC3','PC4', 'PC5', 'PC6','PC7', 'PC8', 'PC9',...
     'PC10', 'PC11','PC12', 'PC13', 'PC14', 'PC15'}; 
%生成表格，按列生成
PC = transpose(PCA1(1:15,:));
result_table=table(tstr,PC(:,1),PC(:,2),PC(:,3),PC(:,4),PC(:,5),PC(:,6),PC(:,7), ...
    PC(:,8), PC(:,9), PC(:,10), PC(:,11), PC(:,12), PC(:,13), PC(:,14), PC(:,15),'VariableNames',col);
%保存表格
writetable(result_table, 'ssta-PCA1-15.csv');

%% ssta = sst-sst_climatlolgy kth modes eof
    [eof_maps2,pc2,V2,expvar2] =  eof_simple(ssta,1000);
    EOF2 = fliplr(V2);
    PCA2 = flipud(pc2);
    %      A2 = (EOF2(:,1:500)*PCA2(1:500,:))'; % Restore the original data field
%     A2 = (EOF2*PCA2)'; % Restore the original data field
%     mask = ~any(isnan(ssta),3);
%     Arestored = nan(size(A2,1),numel(mask));
%     Arestored(:,mask(:)) = A2;
%     Arestored = reshape(Arestored,size(A2,1),size(mask,1),size(mask,2));
%     Arestored = permute(Arestored,[2 3 1]);
%     k = 10000;
%     yval = Arestored(:,:,k);
%     y0yc = sstcli(:,:,doy(tt(k)));
%     ysum = y0yc +Arestored(:,:,k);
%     ydiff = sstyear(:,:,k) - ysum;

%%
kmode = 1;
k = 10957-kmode+1;
figure
subplot(2,2,1)
pcolor(lon,lat,eof_maps1(:,:,k)'); shading flat
colormap(jet)
colorbar
clim([-10 10]*1e-3)
subplot(2,2,2)
plot(PCA1(10957-k+1,:));
subplot(2,2,3)
pcolor(lon,lat,eof_maps2(:,:,kmode)'); shading flat
colormap(jet)
colorbar
clim([-10 10]*1e-3)
subplot(2,2,4)
% plot(PCA2(10957-k+1,:));
plot(PCA2(kmode,:));

