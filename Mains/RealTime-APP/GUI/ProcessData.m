function  [Range_Time_Plot,RANGE_AXIS,Micro_Doppler_Plot,VELOCITY_AXIS] = ProcessData(filename)

    F0 = 77e9; % ��ʼƵ��
    c = physconst('lightspeed'); % ����
    LAMBDA = c / F0; % ����
    D = LAMBDA / 2; % ���߼��
    RX_NUM = 4; % ����������Ŀ
    TX_NUM = 3; % ����������Ŀ
    CHIRP_NUM = 64; % ÿ֡chirp���� �°汾luaΪ128 �ϰ汾64

    IDEL_TIME = 100e-6; % ����ʱ��
    RAMP_TIME = 60e-6; % �������ʱ��
    TC = (IDEL_TIME + RAMP_TIME) * TX_NUM; % ��֡ʱ��
    TF = TC * CHIRP_NUM; % ֡��ʱ��CHIRP

    SLOPE = 60.012e12; % ��Ƶб��
    ADC_SAMPLE = 512; % ADC������
    FS_SAMPLE = 10000e3; % ������
    FRAME_LENGTH = 100; % ֡��
    RANGE_RES = 3e8 / (2 * 1 / FS_SAMPLE * ADC_SAMPLE * SLOPE); % ����ֱ���
    RANGE_AXIS = [1: ADC_SAMPLE] * RANGE_RES; % ���뵥Ԫ
%     MAX_RANGE = 300 * FS_SAMPLE / 1e3  / ( 2 * SLOPE / 1e12 * 1e3);
    VELOCITY_RES = 3e8 / (2 * F0 * TF); % �ٶȷֱ���
%     MAX_VELOCITY = 3e8 / (4 * F0 * TC);  % ����ٶ�
    VELOCITY_AXIS = [-CHIRP_NUM / 2 : CHIRP_NUM / 2 - 1] * VELOCITY_RES; % �ٶȵ�Ԫ

    [data] = readDCA1000(filename);
    INIT_FRAME = 1;
    FRAME_NUM = 100;
    NEW_RX1_DATA = reshape(data(1,:),ADC_SAMPLE,TX_NUM,CHIRP_NUM,FRAME_LENGTH);
%     NEW_RX2_DATA = reshape(data(2,:),ADC_SAMPLE,TX_NUM,CHIRP_NUM,FRAME_LENGTH);
%     NEW_RX3_DATA = reshape(data(3,:),ADC_SAMPLE,TX_NUM,CHIRP_NUM,FRAME_LENGTH);
%     NEW_RX4_DATA = reshape(data(4,:),ADC_SAMPLE,TX_NUM,CHIRP_NUM,FRAME_LENGTH);
%     RADAR_DATA = cat(2,NEW_RX1_DATA,NEW_RX2_DATA,NEW_RX3_DATA,NEW_RX4_DATA); % ��ȡ���ź�����
    RADAR_DATA = NEW_RX1_DATA(:,1,:,:);
    NEW_RADAR_DATA = RADAR_DATA(:,1,:,INIT_FRAME:INIT_FRAME+FRAME_NUM-1);
  
    [RANGE_PROFILE] = RangeFFT(NEW_RADAR_DATA);
    [DOPPLER_PROFILE] = DopplerFFT(RANGE_PROFILE);
    Range_Time_Plot = (abs(sum(squeeze(DOPPLER_PROFILE(:,1,:,:)),2)));
    Micro_Doppler_Plot = (abs(sum(squeeze(DOPPLER_PROFILE(:,1,:,:)),1)));
    
end

