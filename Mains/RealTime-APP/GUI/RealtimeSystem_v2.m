%% ���ļ�Ϊʵʱ����ϵͳ�ڶ���
clc;clear;close all;

run('configure_param.m');
data_name = "trial12";
%% �����ļ���
root_path = 'H:\RadarProcessing\DataFile\TestData\'; % ��·������
data_path = strcat(root_path,data_name); 
mkdir(data_path);

filename = strcat(root_path,data_name,'\adc_data_Raw_0.bin'); % �����ļ������Ƿ����bin�ļ�

tic
SendCaptureCMD(data_name);
while true
    D = dir(filename);
    path_size = D.bytes / 1024; % �ļ���С

    if path_size ~= 0 

       [data] = readDCA1000(filename);  
       max_frame = floor(size(data,2)/(ADC_SAMPLE*TX_NUM*CHIRP_NUM)); 
       RX1_DATA = reshape(data(1,1:ADC_SAMPLE*TX_NUM*CHIRP_NUM*max_frame),ADC_SAMPLE,TX_NUM,CHIRP_NUM,max_frame);
       FRAME_SET = (1:max_frame)*100/1e3;
       [RANGE_PROFILE] = RangeFFT(RX1_DATA(:,1,:,:));
       [DOPPLER_PROFILE] = DopplerFFT(RANGE_PROFILE);
       Range_Time_Plot = (abs(sum(squeeze(DOPPLER_PROFILE(:,1,:,:)),2)));
       Micro_Doppler_Plot = (abs(sum(squeeze(DOPPLER_PROFILE(:,1,:,:)),1)));
       subplot(121);
       imagesc(FRAME_SET,RANGE_AXIS,db(squeeze((Range_Time_Plot))));
       xlabel('Frame Period(s)');ylabel('Range(m)');
       subplot(122);
       imagesc(FRAME_SET,VELOCITY_AXIS,db(squeeze(Micro_Doppler_Plot)));
       xlabel('Frame Period(s)');ylabel('Velocity(m/s)');
       pause(0.01);   
    end

    if path_size == 307200
        break; % ����ļ���С����ɼ�Ҫ�����˳�ѭ��
    end

end
toc
disp('�ɼ����');