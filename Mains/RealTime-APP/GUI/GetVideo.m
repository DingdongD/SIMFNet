clear all; clc
%% ���ļ�����ʵ�ֿ�������ͷ¼��
tic
closepreview;
video_information = imaqhwinfo();
% video_format = video_information.DeviceInfo.SupportedFormats; % �����ڲ鿴��ʽ
% winvideoinfo = imaqhwinfo('winvideo')
vid = videoinput('winvideo',2,'RGB32_1280x720');
preview(vid)
filename = 'E:\RadarProcessing\film';
nframe = 160; % �м�֡����  
nrate = 20; % ÿ��֡�ʱ�ʾÿ�뼸֡��nframe*nrate��ʾ3s
MakeVideo(vid, filename, nframe, nrate, 2) % ��ʼ����ͼ������
toc