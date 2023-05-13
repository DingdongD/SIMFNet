%% Radar Configuration 
%% By Xuliang,22134033@zju.edu.cn
clc;clear;close all;
c = physconst('lightspeed'); % speed of light
f0 = 79e9; % start frequenct
AWR1243_CONFIG.lambda = c / f0; % wave length
AWR1243_CONFIG.D = AWR1243_CONFIG.lambda / 2; % antenna spacing
AWR1243_CONFIG.num_rx = 4; % rx nums
AWR1243_CONFIG.num_tx = 3; % tx nums
AWR1243_CONFIG.chirp_num = 128; % chirp-loop num per frame
AWR1243_CONFIG.adc_sample = 256; % adc samples per chirp
AWR1243_CONFIG.frame_num = 40; % frame nums
idel_time = 30e-6; 
ramp_time = 80e-6; 
TC = (idel_time + ramp_time) * AWR1243_CONFIG.num_tx; % duration per chirp sequence of TDM
TF = TC * AWR1243_CONFIG.chirp_num; % frame time

AWR1243_CONFIG.slope = 46.397e12; % slope frequency
AWR1243_CONFIG.adc_sample = 256; % ADC samples
AWR1243_CONFIG.fs_sample = 6874e3; % sampling rate

range_fft_sample = 256; % adc sample number
doppler_fft_sample = 128; % doppler sample number
frame_fft_sample = 128; % CVD fft number

range_res = 3e8 / (2 * 1 / AWR1243_CONFIG.fs_sample * range_fft_sample * AWR1243_CONFIG.slope); % range resolution
AWR1243_CONFIG.range_axis = [-range_fft_sample / 2 : range_fft_sample / 2 - 1] * range_res / (range_fft_sample / AWR1243_CONFIG.adc_sample); % range index
velocity_res = 3e8 / (2 * f0 * TF); % volicty resolution
AWR1243_CONFIG.velocity_axis = [-doppler_fft_sample / 2 : doppler_fft_sample / 2 - 1] * velocity_res / (doppler_fft_sample / AWR1243_CONFIG.chirp_num); % velocity index
AWR1243_CONFIG.cadence_axis = [-frame_fft_sample / 2 : frame_fft_sample / 2 - 1] * velocity_res / (frame_fft_sample / AWR1243_CONFIG.frame_num); % cadence velocity
AWR1243_CONFIG.range_fft_num = range_fft_sample;
AWR1243_CONFIG.doppler_fft_num = doppler_fft_sample;
AWR1243_CONFIG.frame_fft_num = doppler_fft_sample;
save('AWR1243_CONFIG.mat','AWR1243_CONFIG');