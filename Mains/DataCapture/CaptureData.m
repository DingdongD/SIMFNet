% 第一次需要运行文件 后面需要注释掉
% By Xuliang,22130433@zju.edu.cn
clc;clear;close all;

%% 实际测试需要修改数据变量文件名称
% 文件名称具体见打印出的表格
% wave/swim with buoys/pull with buoys/frolic/freestyle/float with
% buoys/drown/distress/breaststroke/backstroke

% root_path = 'H:\MyDataset\DatasetFile\wave\'; % 根路径名称
% data_name = 'I027';
root_path = 'H:\MyDataset\DatasetFile\'; % 根路径名称
data_name = 'TS2';
data_path = strcat(root_path,data_name);

%% 雷达基本参数配置
F0 = 77e9; % 起始频率
c = physconst('lightspeed'); % 光速
LAMBDA = c / F0; % 波长
D = LAMBDA / 2; % 天线间距
TX_NUM = 3; % 发射天线数目
% 游泳池的实验参数 最远11m 速度3m/s 脚本5
SLOPE = 46.397e12; % 调频斜率
ADC_SAMPLE = 512; % ADC采样点 256
FS_SAMPLE = 6182e3; % 采样率 6847
IDEL_TIME = 30e-6; % 空闲时间
RAMP_TIME = 85e-6; % 脉冲持续时间 80
CHIRP_NUM = 128; % 每帧chirp个数 lua5：128
TC = (IDEL_TIME + RAMP_TIME) * TX_NUM; % 单帧时间
TF = TC * CHIRP_NUM; % 帧间时间CHIRP
RANGE_RES = 3e8 / (2 * 1 / FS_SAMPLE * ADC_SAMPLE * SLOPE); % 距离分辨率
RANGE_AXIS = [-ADC_SAMPLE / 2 : ADC_SAMPLE / 2-1] * RANGE_RES; % 距离单元
%     RANGE_AXIS = [1: ADC_SAMPLE/2] * RANGE_RES; % 距离单元
VELOCITY_RES = 3e8 / (2 * F0 * TF); % 速度分辨率
VELOCITY_AXIS = [-CHIRP_NUM / 2 : CHIRP_NUM / 2 - 1] * VELOCITY_RES; % 速度单元

adc_file_name = strcat(data_path,'\adc_data_Raw_0.bin'); % 检测该文件夹下是否存在bin文件
SendCaptureCMD(data_name); % 发送采集数据的指令
pause(0.1);
FID = fopen(adc_file_name,'r'); % 读取bin文件
rt_show = [];
dt_show = [];
FRAME_SET = [];
init_frame = 0;
while(true)
    D = dir(adc_file_name);
	path_size = D.bytes / 1024; % 文件大小
	if path_size ~= 0 
        NUMADCBITS = 16; % ADC采样精度
        NUMLANES = 4; % 接收天线通道数目，通常不需要改变，除非仅用单通道
        ISREAL = 0; % 0 表示复数，1表示实数

        %% 读取文件并转为有符号数
        
        ADCDATA = fread(FID,'int16');
%         ADCDATA = fread(FID,37498880,'int16');
        % 如果ADC精度为12位/14位则需要对采样数据补偿
        if NUMADCBITS ~= 16
            LMAX = 2^(NUMADCBITS-1)-1;
            ADCDATA(ADCDATA > LMAX) = ADCDATA(ADCDATA > LMAX) - 2^NUMADCBITS;
        end

        %% 参考TI的ADC RAW DATA CAPTURE的PDF 重排通道数据
        if ISREAL
            ADCDATA = reshape(ADCDATA, NUMLANES, []); % 每个接收天线只有一组采样即实部
        else % 每个接收天线有两组采样数据：实部和虚部
            ADCDATA = reshape(ADCDATA, NUMLANES*2, []);
            ADCDATA = ADCDATA([1,2,3,4],:) + sqrt(-1)*ADCDATA([5,6,7,8],:);
        end
        data = ADCDATA(1,:); 
        
        max_frame = floor(size(data,2)/(ADC_SAMPLE*TX_NUM*CHIRP_NUM)); 
        if max_frame ~= 0
            RX1_DATA = reshape(data(1:ADC_SAMPLE*TX_NUM*CHIRP_NUM*max_frame),ADC_SAMPLE,TX_NUM,CHIRP_NUM,max_frame);

            TX1_DATA = squeeze(RX1_DATA(:,1,:,:));
            range_plane = zeros(ADC_SAMPLE/2,max_frame);
            micro_doppler = zeros(CHIRP_NUM,max_frame);
            for frame_idx = 1:max_frame
                adc_data = squeeze(TX1_DATA(:,:,frame_idx));
                adc_data = adc_data - mean(adc_data,1); % 滤除静态杂波
                adc_data = adc_data .* hanning(ADC_SAMPLE); % 加窗
                range_profile = fft(adc_data,ADC_SAMPLE,1); % 距离fft 
                range_profile = range_profile - repmat(mean(range_profile'),size(range_profile,2),1)'; % 滤除速度为0目标
                doppler_profile = fftshift(fft(range_profile,CHIRP_NUM,2),2); % 多普勒fft

                %     cfar_matrix = cfar_ca_2d(doppler_profile,Tr,Td,Gr,Gd,alpha);
                %     doppler_profile(cfar_matrix == 0) = 0;
                dsum = abs(doppler_profile).^2;
                rsum = sum(dsum(1:end/2,:),2);
                vsum = sum(dsum(1:end/2,:),1);
                range_plane(:,frame_idx) = rsum;
                micro_doppler(:,frame_idx) = vsum;
            end
        
            init_frame = init_frame + max_frame;
            FRAME_SET = 1:init_frame;
            axis xy;
            subplot(121);
            rt_show = [rt_show,db(abs(range_plane))/2];
            imagesc(FRAME_SET*100/1e3,RANGE_AXIS(end/2+1:end),rt_show);
            xlabel('Frame Period(s)');ylabel('Range(m)');
            colormap(jet);caxis([80 110]);
            
            axis xy;
            subplot(122);
            dt_show = [dt_show,(db(abs(micro_doppler)))/2];
            imagesc(FRAME_SET*100/1e3,VELOCITY_AXIS,dt_show);
            xlabel('Frame Period(s)');ylabel('Velocity(m/s)');
            colormap(jet);caxis([80 110]);
            pause(0.01); 
        end
    end
    if path_size == 460800/10
        fclose(FID); % 关闭bin文件
        break; % 如果文件大小满足采集要求则退出循环
    end
end


