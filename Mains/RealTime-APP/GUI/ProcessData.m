function  [Range_Time_Plot,RANGE_AXIS,Micro_Doppler_Plot,VELOCITY_AXIS] = ProcessData(filename)

    F0 = 77e9; % 起始频率
    c = physconst('lightspeed'); % 光速
    LAMBDA = c / F0; % 波长
    D = LAMBDA / 2; % 天线间距
    RX_NUM = 4; % 接收天线数目
    TX_NUM = 3; % 发射天线数目
    CHIRP_NUM = 64; % 每帧chirp个数 新版本lua为128 老版本64

    IDEL_TIME = 100e-6; % 空闲时间
    RAMP_TIME = 60e-6; % 脉冲持续时间
    TC = (IDEL_TIME + RAMP_TIME) * TX_NUM; % 单帧时间
    TF = TC * CHIRP_NUM; % 帧间时间CHIRP

    SLOPE = 60.012e12; % 调频斜率
    ADC_SAMPLE = 512; % ADC采样点
    FS_SAMPLE = 10000e3; % 采样率
    FRAME_LENGTH = 100; % 帧数
    RANGE_RES = 3e8 / (2 * 1 / FS_SAMPLE * ADC_SAMPLE * SLOPE); % 距离分辨率
    RANGE_AXIS = [1: ADC_SAMPLE] * RANGE_RES; % 距离单元
%     MAX_RANGE = 300 * FS_SAMPLE / 1e3  / ( 2 * SLOPE / 1e12 * 1e3);
    VELOCITY_RES = 3e8 / (2 * F0 * TF); % 速度分辨率
%     MAX_VELOCITY = 3e8 / (4 * F0 * TC);  % 最大速度
    VELOCITY_AXIS = [-CHIRP_NUM / 2 : CHIRP_NUM / 2 - 1] * VELOCITY_RES; % 速度单元

    [data] = readDCA1000(filename);
    INIT_FRAME = 1;
    FRAME_NUM = 100;
    NEW_RX1_DATA = reshape(data(1,:),ADC_SAMPLE,TX_NUM,CHIRP_NUM,FRAME_LENGTH);
%     NEW_RX2_DATA = reshape(data(2,:),ADC_SAMPLE,TX_NUM,CHIRP_NUM,FRAME_LENGTH);
%     NEW_RX3_DATA = reshape(data(3,:),ADC_SAMPLE,TX_NUM,CHIRP_NUM,FRAME_LENGTH);
%     NEW_RX4_DATA = reshape(data(4,:),ADC_SAMPLE,TX_NUM,CHIRP_NUM,FRAME_LENGTH);
%     RADAR_DATA = cat(2,NEW_RX1_DATA,NEW_RX2_DATA,NEW_RX3_DATA,NEW_RX4_DATA); % 获取重排后数据
    RADAR_DATA = NEW_RX1_DATA(:,1,:,:);
    NEW_RADAR_DATA = RADAR_DATA(:,1,:,INIT_FRAME:INIT_FRAME+FRAME_NUM-1);
  
    [RANGE_PROFILE] = RangeFFT(NEW_RADAR_DATA);
    [DOPPLER_PROFILE] = DopplerFFT(RANGE_PROFILE);
    Range_Time_Plot = (abs(sum(squeeze(DOPPLER_PROFILE(:,1,:,:)),2)));
    Micro_Doppler_Plot = (abs(sum(squeeze(DOPPLER_PROFILE(:,1,:,:)),1)));
    
end

