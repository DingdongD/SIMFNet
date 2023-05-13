
%% ���ļ�����ʵ��׼ʵʱ����ϵͳ
clc;clear;close all;
data_name = "trial8";
%% �����ļ���
root_path = 'H:\RadarProcessing\DataFile\TestData\'; % ��·������
data_path = strcat(root_path,data_name); 
mkdir(data_path);

TOTAL_FRAME_DURATION = 400; % ֡��Ϊ200֡
FRAME_INTERVAL = 100; % �ű��ļ�����Ϊ20֡
filename = strcat(root_path,data_name,'\adc_data_Raw_0.bin'); % �����ļ������Ƿ����bin�ļ�
FRAME_AXIS = (1:TOTAL_FRAME_DURATION)*100/1e3; % ֡ʱ��
file_path = strcat(root_path,data_name,'\');

RT_SET = [];
DT_SET = [];

figure(1);
tic 
for frame_index = 1:TOTAL_FRAME_DURATION/FRAME_INTERVAL
    % ����ָ��
    SendCaptureCMD(data_name);
    path_size = 0; % ��ʼ��bin�ļ���С
    % ��ȡ�������ļ�
    while(path_size ~= 76800*2) % ������ļ��в������򴴽�һ����Ŀ¼
       D = dir(filename);
       path_size = D.bytes / 1024; % �ļ���С
       disp('�ļ����ڲɼ�');
    end
    [Range_Time_Plot,RANGE_AXIS,Micro_Doppler_Plot,VELOCITY_AXIS] = ProcessData(filename);
    RT_SET = cat(2,RT_SET,squeeze(Range_Time_Plot));
    DT_SET = cat(2,DT_SET,squeeze(Micro_Doppler_Plot));
    FRAME_SET = FRAME_AXIS(1+FRAME_INTERVAL*(frame_index-1):FRAME_INTERVAL*frame_index);
    subplot(121);
    imagesc(FRAME_SET,RANGE_AXIS,db(squeeze((RT_SET)))); % RANGE_AXIS,
    xlabel('Frame Period(s)');ylabel('Range(m)');
    subplot(122);
    imagesc(FRAME_SET,VELOCITY_AXIS,db(squeeze(DT_SET)));
    xlabel('Frame Period(s)');ylabel('Velocity(m/s)');
    pause(0.01)
    change_name = strcat(file_path,'adc_data_Raw_0',num2str(frame_index),'.bin');
    movefile(filename,change_name);
end
toc


