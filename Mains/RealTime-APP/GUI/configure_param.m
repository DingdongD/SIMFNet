%% 雷达基本参数的配置文件
%% BY YUXULIANG,ZJU,20220216
%% Radar Parameter
% clc;clear;close all
F0 = 77e9; % 起始频率
c = physconst('lightspeed'); % 光速
LAMBDA = c / F0; % 波长
D = LAMBDA / 2; % 天线间距
RX_NUM = 4; % 接收天线数目
TX_NUM = 3; % 发射天线数目
CHIRP_NUM = 128; % 每帧chirp个数 新版本lua为128 老版本64

IDEL_TIME = 30e-6; % 空闲时间
RAMP_TIME = 80e-6; % 脉冲持续时间
TC = (IDEL_TIME + RAMP_TIME) * TX_NUM; % 单帧时间
TF = TC * CHIRP_NUM; % 帧间时间CHIRP

SLOPE = 48.521e12; % 调频斜率
ADC_SAMPLE = 256; % ADC采样点
FS_SAMPLE = 3594e3; % 采样率

FRAME_LENGTH = 200; % 帧数 新版本lua为100 老版本200
RANGE_RES = 3e8 / (2 * 1 / FS_SAMPLE * ADC_SAMPLE * SLOPE); % 距离分辨率
RANGE_AXIS = [-ADC_SAMPLE / 2 : ADC_SAMPLE / 2-1] * RANGE_RES; % 距离单元
% RANGE_AXIS = [1: ADC_SAMPLE] * RANGE_RES; % 距离单元
MAX_RANGE = 300 * FS_SAMPLE / 1e3  / ( 2 * SLOPE / 1e12 * 1e3);
VELOCITY_RES = 3e8 / (2 * F0 * TF); % 速度分辨率
MAX_VELOCITY = 3e8 / (4 * F0 * TC);  % 最大速度
VELOCITY_AXIS = [-CHIRP_NUM / 2 : CHIRP_NUM / 2 - 1] * VELOCITY_RES; % 速度单元
ANGLE_MAX = (asin(LAMBDA/(2*D)));  % 最大测角
ANGLE_RES = 2 / RX_NUM / TX_NUM * 180 / pi; % 角度分辨率 单位为rad 通过*180/pi实现°的转换
ANGLE_SAMPLE = 128; % 角度采样率
ANGLE_AXIS = [-ANGLE_MAX:pi/(ANGLE_SAMPLE-1):ANGLE_MAX]; % 角度索引
% ANGLE_AXIS = [-pi:2*pi/(ANGLE_SAMPLE-1):pi]; % 角度索引
% ANGLE_AXIS = ANGLE_AXIS(1:end-1);
