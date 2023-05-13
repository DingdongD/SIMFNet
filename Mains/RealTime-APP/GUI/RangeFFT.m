function [RANGE_PROFILE] = RangeFFT(NewData)
%% ����FFT�������ļ�
%% NewData : ���ź����������
%% RangeData: ����FFT�������
%% BY YUXULIANG,ZJU,20211110

    %% ��ȡ��ز���
    ADC_SAMPLE = size(NewData,1); % ADC��������
    VIRTUAL_NUM = size(NewData,2); % ����������Ŀ
    CHIRP_NUM = size(NewData,3); % ÿ֡Chirp��Ŀ
    FRAME_LENGTH = size(NewData,4); % ֡����
    RANGE_WIN = hamming(ADC_SAMPLE); % ����ά����

    %% ����FFT
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