function [DOPPLER_PROFILE] = DopplerFFT(RANGE_PROFILE)
%% 多普勒FFT的配置文件
%% RANGE_PROFILE : 距离FFT后的数据
%% DOPPLER_PROFILE: 多普勒FFT后的数据
%% BY YUXULIANG,ZJU,20211110

   %% 获取相关参数
    ADC_SAMPLE = size(RANGE_PROFILE,1); % ADC采样点数
    VIRTUAL_NUM = size(RANGE_PROFILE,2); % 虚拟天线数目
    CHIRP_NUM = size(RANGE_PROFILE,3); % 每帧Chirp数目
    FRAME_LENGTH = size(RANGE_PROFILE,4); % 帧长度
    DOPPLER_WIN = hamming(CHIRP_NUM); % 距离维窗口

    %% 多普勒FFT
    DOPPLER_PROFILE = fftshift(fft(RANGE_PROFILE,128,3),3);
%     for frame_index = 1:FRAME_LENGTH
%         for rtx_index = 1:VIRTUAL_NUM
%             for adc_index = 1:ADC_SAMPLE
%                 TEMP = squeeze(RANGE_PROFILE(adc_index,rtx_index,:,frame_index)).* (DOPPLER_WIN);
%                 TEMP_FFT = fftshift(fft(TEMP));
%                 DOPPLER_PROFILE(adc_index,rtx_index,:,frame_index) = TEMP_FFT;
%             end
%         end
%     end
end