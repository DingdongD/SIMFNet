%% ���ļ����ڲ���Range-Azimuth-Doppler�����µ��źŴ���ջ
%% By Xuliang,20230414
clc;clear;close all;

dataPath = './dataset/adc_raw_dat.mat'; % �����ļ�·��
ShowIQ = 1; % �Ƿ���ʾIQ�ź�
ShowRange = 1; % �Ƿ���ʾRangePorfile
ShowRA = 1; % �Ƿ���ʾRAM
ShowRD = 1; % �Ƿ���ʾRDM
ShowCFAR = 1; % �Ƿ���ʾCFAR���
ShowPeak = 1; % �Ƿ���ʾ�ۺϷ�ֵ���
IQFlag = 1; % �Ƿ�ѡ��IQ·�ź� 0-��· 1-˫·
saveFlag = 0; % �Ƿ񱣴��ļ�

%% Ŀ���ϵͳ��������
disp(strcat(['=====','ˢ��Ŀ���ϵͳ����','====='])); % �����״�ģʽ
tarOut = ConfigureTarget;     % ����Ŀ����Ϣ
cfgOut = ConfigureParameter;  % ���ɺ��ײ��״�ϵͳ����

%% �״�ɼ�����
disp(strcat(['=====','�״���빤��״̬','=====']));     
[RawData] = GenerateAdcData(tarOut, cfgOut, IQFlag, saveFlag, dataPath); % ��ʼģ�������

%% �����������
c = physconst('LightSpeed'); % ����
fc = cfgOut.fc; % ��Ƶ Hz
lambda = c / fc; % ����

ADCNum = cfgOut.ADCNum; % ADC������Ŀ
ChirpNum = cfgOut.ChirpNum; % ÿ֡����Chirp��Ŀ

numTx = cfgOut.numTx; % ����������Ŀ
numRx = cfgOut.numRx; % ����������Ŀ
% arr = cfgOut.array; % ��Ԫ����[���װ汾������]
virtual_array = cfgOut.virtual_array; % ��������struct
arr = virtual_array.virtual_arr;

arrNum = numTx * numRx; % ��Ԫ��Ŀ
arrDx = cfgOut.arrdx; % ��λ����Ԫ���
arrDy = cfgOut.arrdx; % ��������Ԫ���

validB = cfgOut.validB; % ��Ч����
range_res = c / (2 * validB); % ����ֱ���

TF = cfgOut.Tc * (numTx * ChirpNum);
doppler_res = lambda / (2 * TF); % �����շֱ���
Frame = cfgOut.Frame; % ֡��

% �����������ٶ�����
if IQFlag
    velocityIndex = [-ChirpNum / 2 : 1 : ChirpNum / 2 - 1] * doppler_res;
    rangeIndex = (0 : ADCNum - 1) * range_res;
else
    velocityIndex = [-ChirpNum / 2 : 1 : ChirpNum / 2 - 1] * doppler_res;
    rangeIndex = (-ADCNum / 2 : 1 : ADCNum / 2 - 1) * range_res;
    rangeIndex = rangeIndex(end/2+1:end);  % ȡ����һ��
end

%% IQƽ�����
if ShowIQ
    if IQFlag % IQ��·�ź�
        figure(1);
        set(gcf,'unit','centimeters','position',[10,12,10,10])
        plot(1:ADCNum, real(RawData(:,1,1))); hold on;
        plot(1:ADCNum, imag(RawData(:,1,1))); hold off;
        xlabel('ADC Points');ylabel('Amplitude');title('IQ-signal Analysis');
        legend('I-Chain','Q-Chain');grid minor;
    else
        figure(1);
        set(gcf,'unit','centimeters','position',[10,12,10,10])
        plot(1:ADCNum, (RawData(:,1,1)));
        xlabel('ADC Points');ylabel('Amplitude');title('IQ-signal Analysis');grid minor;
    end
end
RawData = reshape(RawData, ADCNum, ChirpNum, [], arrNum);
disp(strcat(['�״��źŵ�ά��Ϊ��',num2str(size(RawData))]));


for frame_id = 1 : Frame
    adcData = squeeze(RawData(:, :, frame_id, :));
    
    %% ����άFFT
    disp(strcat(['=====','Range-Profile����','====='])); 
    tic
    fftOut1 = rangeFFT(adcData, IQFlag);
    toc
    rangeFFTOut = fftOut1.rangeFFT;
    
    if ShowRange
        figure(2);
        set(gcf,'unit','centimeters','position',[20,12,10,10])
        plot(rangeIndex, db(rangeFFTOut(:,1,1))); 
        xlabel('Range(m)');ylabel('Amplitude(dB)'); 
        title(strcat(['��',num2str(frame_id),'֡-Ŀ�����ֲ�']));grid minor;  
        pause(0.1);
    end
    
    %% ����-��λͼ����
    cfgDOA.FFTNum = 180; % FFT����
    cfgDOA.AziMethod = 'MUSIC'; % ���ڱ��׼ܹ�Ŀ�������ö������Ϣ ��֧�ֳ����㷨
    cfgDOA.AzisigNum = 1; % Լ��ÿ��CELL�ϵ���Դ��Ŀ
    cfgDOA.thetaGrids = linspace(-90, 90, cfgDOA.FFTNum); % �ռ�����
    
    disp(strcat(['=====','Range-Azimuth-Map����','====='])); 
    tic
    RAM = GenerateRAM(rangeFFTOut, cfgOut, cfgDOA);
    toc
    if ShowRA
        figure(3);
        set(gcf,'unit','centimeters','position',[30,12,10,10])
        [theta,rho] = meshgrid(cfgDOA.thetaGrids, rangeIndex); % ����
        xaxis = rho .* cosd(theta); % ������
        yaxis = rho .* sind(theta); % ������
        surf(yaxis,xaxis,db(abs(RAM)),'EdgeColor','none'); 
        view(2);colormap('jet');
        xlabel('Range(m)','fontsize',15,'fontname','Times New Roman');ylabel('Range(m)','fontsize',15,'fontname','Times New Roman');grid on; axis xy
        title(strcat(['��',num2str(frame_id),'֡-���뷽λ��ͼ']));grid minor;  
        set(gca,'GridLineStyle','- -');
        set(gca,'GridAlpha',0.2);
        set(gca,'LineWidth',1.5);
        set(gca,'xminortick','on'); 
        set(gca,'ygrid','on','GridColor',[0 0 0]);
        colorbar;
    end
    
    %% CFAR���
    disp(strcat(['=====','���龯���','====='])); 
    Pfa = 1e-2; % �龯����
    TestCells = [8, 8]; % �ο���
    GuardCells = [4, 4]; % ������
    
    tic
    [cfarOut] = CFAR_2D(abs(RAM).^2, Pfa, TestCells, GuardCells);
    toc
    cfarMap = cfarOut.cfarMap; % �������
    noiseOut = cfarOut.noiseOut; % �������
    snrOut = cfarOut.snrOut; % �������� RAM������Ȳ��ܱ�ʾ��ʵ�����(���ֿ��׷�����)
    
    if ShowCFAR
        figure(4);
        set(gcf,'unit','centimeters','position',[30,0,10,10]);
        imagesc(cfgDOA.thetaGrids, rangeIndex, cfarMap);
        ylabel('Range(m)');xlabel('Angle(deg)'); colormap('jet');
        title(strcat(['��',num2str(frame_id),'֡Ŀ��-CFAR�����']));
        grid minor;axis xy;
        pause(0.1);
    end
    
    %% ��ֵ�ۺ�-��ȡ��Ŀ��
    disp(strcat(['=====','��ֵ�ۺ�','====='])); 
    [range_idx, doppler_idx] = find(cfarMap);
    cfar_out_idx = [range_idx doppler_idx]; % ��ȡCFAR����������������
    tic
    [rd_peak_list, rd_peak] = peakFocus(db(RAM), cfar_out_idx);
    toc
    peakMap = zeros(size(cfarMap)); % ��ֵ�ۺϽ������
    for peak_idx = 1 :size(rd_peak_list, 2)
        peakMap(rd_peak_list(1,peak_idx), rd_peak_list(2,peak_idx)) = 1;
    end
    
    if ShowPeak
        figure(5);
        set(gcf,'unit','centimeters','position',[40,12,10,10])
        imagesc(cfgDOA.thetaGrids, rangeIndex, peakMap);
        ylabel('Range(m)');xlabel('Angle(deg)'); colormap('jet');
        title(strcat(['��',num2str(frame_id),'֡Ŀ��-��ֵ�ۺϽ��']));
        grid minor;axis xy;xlim([-60 60]);
        pause(0.1);
    end
    
    % ������Ҫ����CFAR����Ŀ���[�洢��⵽��Ŀ�������������λ����]
    targetPerFrame = {}; 
    targetPerFrame.rangeSet = []; % �洢����
    targetPerFrame.velocitySet = []; % �洢�ٶ�[Ϊ�����ʼ��]
    targetPerFrame.azimuthSet = []; % �洢��λ
    
    [cfar_rid, cfar_cid] = find(cfarMap); % rid��ʾ���뵥Ԫ���� cid��ʾ�Ƕȵ�Ԫ����
    targetPerFrame.rangeSet = [targetPerFrame.rangeSet, rangeIndex(cfar_rid)];
    targetPerFrame.azimuthSet = [targetPerFrame.azimuthSet, cfgDOA.thetaGrids(cfar_cid)];
    
    %% ������FFT����[�ʹ�ͳ���̲�һ��������ط���Ҫʹ�û�������rangeFFT]
    fftOut2 = dopplerFFT(rangeFFTOut); % �������ݴ�СΪ256*128*12
    dopplerFFTOut = fftOut2.dopplerFFT; % ��ȡ������άFFT���� 
    
    %% ����ɻ���
    disp(strcat(['=====','����ɻ���','====='])); 
    RDM = dopplerFFTOut;
    tic
    [accumulateRD] = incoherent_accumulation(RDM);
    toc
    
    if ShowRD
        figure(6);
        set(gcf,'unit','centimeters','position',[20,0,10,10])
        imagesc(rangeIndex, velocityIndex, db(dopplerFFTOut(:,:,1).'));
        xlabel('Range(m)');ylabel('Velocity(m/s)'); colormap('jet');
        title(strcat(['��',num2str(frame_id),'֡Ŀ��-��������շֲ�']));
        grid minor; axis xy;
        pause(0.1);
    end
    
    for tarRangeIdx = 1 : length(targetPerFrame.rangeSet) % ������⵽�ľ��뵥Ԫ �����׷�����ٶ�
        [velVal, velIdx] = findpeaks(accumulateRD(cfar_rid(tarRangeIdx),:)); % �����ٶ��׷�
        [sortVal, sortIdx] = sort(velVal); % �Է�ֵ��������
        Idx = velIdx(sortIdx(end)); % ѡ�����ֵ
        targetPerFrame.velocitySet = [targetPerFrame.velocitySet, velocityIndex(Idx)];
    end
    
end