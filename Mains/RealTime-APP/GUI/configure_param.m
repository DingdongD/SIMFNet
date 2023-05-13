%% �״���������������ļ�
%% BY YUXULIANG,ZJU,20220216
%% Radar Parameter
% clc;clear;close all
F0 = 77e9; % ��ʼƵ��
c = physconst('lightspeed'); % ����
LAMBDA = c / F0; % ����
D = LAMBDA / 2; % ���߼��
RX_NUM = 4; % ����������Ŀ
TX_NUM = 3; % ����������Ŀ
CHIRP_NUM = 128; % ÿ֡chirp���� �°汾luaΪ128 �ϰ汾64

IDEL_TIME = 30e-6; % ����ʱ��
RAMP_TIME = 80e-6; % �������ʱ��
TC = (IDEL_TIME + RAMP_TIME) * TX_NUM; % ��֡ʱ��
TF = TC * CHIRP_NUM; % ֡��ʱ��CHIRP

SLOPE = 48.521e12; % ��Ƶб��
ADC_SAMPLE = 256; % ADC������
FS_SAMPLE = 3594e3; % ������

FRAME_LENGTH = 200; % ֡�� �°汾luaΪ100 �ϰ汾200
RANGE_RES = 3e8 / (2 * 1 / FS_SAMPLE * ADC_SAMPLE * SLOPE); % ����ֱ���
RANGE_AXIS = [-ADC_SAMPLE / 2 : ADC_SAMPLE / 2-1] * RANGE_RES; % ���뵥Ԫ
% RANGE_AXIS = [1: ADC_SAMPLE] * RANGE_RES; % ���뵥Ԫ
MAX_RANGE = 300 * FS_SAMPLE / 1e3  / ( 2 * SLOPE / 1e12 * 1e3);
VELOCITY_RES = 3e8 / (2 * F0 * TF); % �ٶȷֱ���
MAX_VELOCITY = 3e8 / (4 * F0 * TC);  % ����ٶ�
VELOCITY_AXIS = [-CHIRP_NUM / 2 : CHIRP_NUM / 2 - 1] * VELOCITY_RES; % �ٶȵ�Ԫ
ANGLE_MAX = (asin(LAMBDA/(2*D)));  % �����
ANGLE_RES = 2 / RX_NUM / TX_NUM * 180 / pi; % �Ƕȷֱ��� ��λΪrad ͨ��*180/piʵ�֡��ת��
ANGLE_SAMPLE = 128; % �ǶȲ�����
ANGLE_AXIS = [-ANGLE_MAX:pi/(ANGLE_SAMPLE-1):ANGLE_MAX]; % �Ƕ�����
% ANGLE_AXIS = [-pi:2*pi/(ANGLE_SAMPLE-1):pi]; % �Ƕ�����
% ANGLE_AXIS = ANGLE_AXIS(1:end-1);
