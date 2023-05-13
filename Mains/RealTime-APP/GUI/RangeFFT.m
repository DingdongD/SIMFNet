function [RANGE_PROFILE] = RangeFFT(NewData)
%% 距离FFT的配置文件
%% NewData : 重排后的数据阵列
%% RangeData: 距离FFT后的数据
%% BY YUXULIANG,ZJU,20211110

    %% 获取相关参数
    ADC_SAMPLE = size(NewData,1); % ADC采样点数
    VIRTUAL_NUM = size(NewData,2); % 虚拟天线数目
    CHIRP_NUM = size(NewData,3); % 每帧Chirp数目
    FRAME_LENGTH = size(NewData,4); % 帧长度
    RANGE_WIN = hamming(ADC_SAMPLE); % 距离维窗口

    %% 距离FFT
    RANGE_PROFILE = [];
    RANGE_PROFILE = fftshift(fft(NewData,256,1),1);
%     for frame_index = 1:FRAME_LENGTH
%         for rtx_index = 1:VIRTUAL_NUM
%             for chirp_index = 1:CHIRP_NUM
%                 TEMP = squeeze(NewData(:,rtx_index,chirp_index,frame_index)) .* RANGE_WIN;
%                 TEMP_FFT = fftshift(fft(TEMP));
%                 RANGE_PROFILE(:,rtx_index,chirp_index,frame_index) = TEMP_FFT;    
%             end
%         end
%     end
% end