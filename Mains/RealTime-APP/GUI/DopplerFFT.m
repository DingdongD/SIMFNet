function [DOPPLER_PROFILE] = DopplerFFT(RANGE_PROFILE)
%% ������FFT�������ļ�
%% RANGE_PROFILE : ����FFT�������
%% DOPPLER_PROFILE: ������FFT�������
%% BY YUXULIANG,ZJU,20211110

   %% ��ȡ��ز���
    ADC_SAMPLE = size(RANGE_PROFILE,1); % ADC��������
    VIRTUAL_NUM = size(RANGE_PROFILE,2); % ����������Ŀ
    CHIRP_NUM = size(RANGE_PROFILE,3); % ÿ֡Chirp��Ŀ
    FRAME_LENGTH = size(RANGE_PROFILE,4); % ֡����
    DOPPLER_WIN = hamming(CHIRP_NUM); % ����ά����

    %% ������FFT
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