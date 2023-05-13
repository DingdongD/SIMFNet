% ��һ����Ҫ�����ļ� ������Ҫע�͵�
% By Xuliang,22130433@zju.edu.cn
clc;clear;close all;

%% ʵ�ʲ�����Ҫ�޸����ݱ����ļ�����
% �ļ����ƾ������ӡ���ı��
% wave/swim with buoys/pull with buoys/frolic/freestyle/float with
% buoys/drown/distress/breaststroke/backstroke

% root_path = 'H:\MyDataset\DatasetFile\wave\'; % ��·������
% data_name = 'I027';
root_path = 'H:\MyDataset\DatasetFile\'; % ��·������
data_name = 'TS2';
data_path = strcat(root_path,data_name);

%% �״������������
F0 = 77e9; % ��ʼƵ��
c = physconst('lightspeed'); % ����
LAMBDA = c / F0; % ����
D = LAMBDA / 2; % ���߼��
TX_NUM = 3; % ����������Ŀ
% ��Ӿ�ص�ʵ����� ��Զ11m �ٶ�3m/s �ű�5
SLOPE = 46.397e12; % ��Ƶб��
ADC_SAMPLE = 512; % ADC������ 256
FS_SAMPLE = 6182e3; % ������ 6847
IDEL_TIME = 30e-6; % ����ʱ��
RAMP_TIME = 85e-6; % �������ʱ�� 80
CHIRP_NUM = 128; % ÿ֡chirp���� lua5��128
TC = (IDEL_TIME + RAMP_TIME) * TX_NUM; % ��֡ʱ��
TF = TC * CHIRP_NUM; % ֡��ʱ��CHIRP
RANGE_RES = 3e8 / (2 * 1 / FS_SAMPLE * ADC_SAMPLE * SLOPE); % ����ֱ���
RANGE_AXIS = [-ADC_SAMPLE / 2 : ADC_SAMPLE / 2-1] * RANGE_RES; % ���뵥Ԫ
%     RANGE_AXIS = [1: ADC_SAMPLE/2] * RANGE_RES; % ���뵥Ԫ
VELOCITY_RES = 3e8 / (2 * F0 * TF); % �ٶȷֱ���
VELOCITY_AXIS = [-CHIRP_NUM / 2 : CHIRP_NUM / 2 - 1] * VELOCITY_RES; % �ٶȵ�Ԫ

adc_file_name = strcat(data_path,'\adc_data_Raw_0.bin'); % �����ļ������Ƿ����bin�ļ�
SendCaptureCMD(data_name); % ���Ͳɼ����ݵ�ָ��
pause(0.1);
FID = fopen(adc_file_name,'r'); % ��ȡbin�ļ�
rt_show = [];
dt_show = [];
FRAME_SET = [];
init_frame = 0;
while(true)
    D = dir(adc_file_name);
	path_size = D.bytes / 1024; % �ļ���С
	if path_size ~= 0 
        NUMADCBITS = 16; % ADC��������
        NUMLANES = 4; % ��������ͨ����Ŀ��ͨ������Ҫ�ı䣬���ǽ��õ�ͨ��
        ISREAL = 0; % 0 ��ʾ������1��ʾʵ��

        %% ��ȡ�ļ���תΪ�з�����
        
        ADCDATA = fread(FID,'int16');
%         ADCDATA = fread(FID,37498880,'int16');
        % ���ADC����Ϊ12λ/14λ����Ҫ�Բ������ݲ���
        if NUMADCBITS ~= 16
            LMAX = 2^(NUMADCBITS-1)-1;
            ADCDATA(ADCDATA > LMAX) = ADCDATA(ADCDATA > LMAX) - 2^NUMADCBITS;
        end

        %% �ο�TI��ADC RAW DATA CAPTURE��PDF ����ͨ������
        if ISREAL
            ADCDATA = reshape(ADCDATA, NUMLANES, []); % ÿ����������ֻ��һ�������ʵ��
        else % ÿ����������������������ݣ�ʵ�����鲿
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
                adc_data = adc_data - mean(adc_data,1); % �˳���̬�Ӳ�
                adc_data = adc_data .* hanning(ADC_SAMPLE); % �Ӵ�
                range_profile = fft(adc_data,ADC_SAMPLE,1); % ����fft 
                range_profile = range_profile - repmat(mean(range_profile'),size(range_profile,2),1)'; % �˳��ٶ�Ϊ0Ŀ��
                doppler_profile = fftshift(fft(range_profile,CHIRP_NUM,2),2); % ������fft

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
        fclose(FID); % �ر�bin�ļ�
        break; % ����ļ���С����ɼ�Ҫ�����˳�ѭ��
    end
end


