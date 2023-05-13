% By Xuliang,22134033@zju.edu.cn
% FUNC:Batch folder processing of radar data for each aquactic human activity is performed to generate range-time maps, 
% Doppler-time maps, cadence velocity diagrams and SVM feature extraction datasets.

clc;clear;close all;
data_path = 'H:\RadarProcessing\DataFile\PaperData\';
act = 'freestyle';
real_data_path = strcat(data_path,act,'\');
files = dir(real_data_path);

f0 = 77e9; 
c = physconst('lightspeed'); 
lambda = c / f0; 
D = lambda / 2; 
num_rx = 4; 
num_tx = 3; 
chirp_num = 128; 

idel_time = 30e-6; 
ramp_time = 80e-6; 
TC = (idel_time + ramp_time) * num_tx; 
TF = TC * chirp_num;
frame_length = 200; 
overlap = 0.25;
slope = 46.397e12;
adc_sample = 256; 
fs_sample = 6874e3; 
range_res = 3e8 / (2 * 1 / fs_sample * adc_sample * slope); 
range_fft_sample = 256; 
doppler_fft_sample = 128; 
% range_axis = [-range_fft_sample / 2 : range_fft_sample / 2 - 1] * range_res / (range_fft_sample / adc_sample); 
range_axis = [1 : range_fft_sample] * range_res / (range_fft_sample / adc_sample); 
velocity_res = 3e8 / (2 * f0 * TF); 
velocity_axis = [-doppler_fft_sample / 2 : doppler_fft_sample / 2 - 1] * velocity_res / (doppler_fft_sample / chirp_num);
frame_axis = (1:frame_length/overlap)*100/1000*overlap;

for idx = 3:length(files)
    file_path = strcat(real_data_path,files(idx).name,'\');
    son_file = dir(file_path);
    bin_path = strcat(file_path,son_file(4).name);
    [ADC_DATA] = readDCA1000(bin_path);
    adc_data = ADC_DATA(1,:);
    max_frame = round(length(adc_data)/adc_sample/chirp_num/num_tx);
    adc_data = adc_data(1,1:max_frame*adc_sample*chirp_num*num_tx);
    min_range = 5;max_range = 100;
    for tx_idx = 1 : 1
        
        [rt_plot,ud_plot] = data_stft(adc_data,adc_sample,num_tx,tx_idx,chirp_num,max_frame,overlap,min_range,max_range,range_axis);
        db_rt_plot = db(rt_plot(min_range:max_range,:)+eps)/2;
        db_ud_plot = db(ud_plot+eps)/2;
        init_idx = 1;end_idx = 80;interval_idx = 80;
        
        for fra_idx = 1:floor((frame_length / overlap - end_idx)/interval_idx)
            rt_data = db_rt_plot(:,init_idx+interval_idx*(fra_idx-1):end_idx+interval_idx*(fra_idx-1));
            ud_data = db_ud_plot(:,init_idx+interval_idx*(fra_idx-1):end_idx+interval_idx*(fra_idx-1));
            
            
            cvd_plot = fftshift(fft(ud_data,size(ud_data,2),2),2);
            cvd_data = db(cvd_plot)/2;

            figure(1);
            imagesc(rt_data);colormap('jet');caxis([80 110]);axis xy;
            set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w');
            set(gcf,'position',[1500,400, 224, 224]);
            set(gca,'looseInset',[0 0 0 0]);
            axis off
            f=getframe(gcf);
            imwrite(f.cdata,['H:\RadarProcessing\Data\Origin\',act,'\rt\',strcat(files(idx).name,'_',num2str(tx_idx-1),num2str(fra_idx)),'.jpg'])            
            saveas(1,['H:\RadarProcessing\Data\Origin\',act,'\rt\',strcat(files(idx).name,'_',num2str(tx_idx-1),num2str(fra_idx)),'.jpg']);

            figure(2);
            imagesc(ud_data);colormap('jet');caxis([80 110]);axis xy;
            set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w');
            set(gcf,'position',[1500,400, 224, 224]);
            set(gca,'looseInset',[0 0 0 0]);
            axis off
            f=getframe(gcf);
            imwrite(f.cdata,['H:\RadarProcessing\Data\Origin\',act,'\md\',strcat(files(idx).name,'_',num2str(tx_idx-1),num2str(fra_idx)),'.jpg'])
            saveas(2,['H:\RadarProcessing\Data\Origin\',act,'\md\',strcat(files(idx).name,'_',num2str(tx_idx-1),num2str(fra_idx)),'.jpg']);

            figure(3);
            imagesc(cvd_data);colormap('jet');caxis([0 30]);axis xy;
            set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w');
            set(gcf,'position',[1500,400, 224, 224]);
            set(gca,'looseInset',[0 0 0 0]);
            axis off
            f=getframe(gcf);
            imwrite(f.cdata,['H:\RadarProcessing\Data\Origin\',act,'\cvd\',strcat(files(idx).name,'_',num2str(tx_idx-1),num2str(fra_idx)),'.jpg'])
            saveas(3,['H:\RadarProcessing\Data\Origin\',act,'\cvd\',strcat(files(idx).name,'_',num2str(tx_idx-1),num2str(fra_idx)),'.jpg']);
            
            [md_feature,cfs_feature,rt_feature] = extract_feature(velocity_axis,ud_data,rt_data,cvd_data,size(ud_data,2));
            feature_set = [md_feature,cfs_feature,rt_feature];
            data_name = strcat('H:\RadarProcessing\Data\Origin\',act,'\svm\',files(idx).name,'_',num2str(tx_idx-1),num2str(fra_idx),'.mat');
            save(data_name,'feature_set');
        end
    end
end



