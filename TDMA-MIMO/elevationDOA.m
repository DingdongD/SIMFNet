function [doaOut] = elevationDOA(arrData, cfgDOA)
    %% ���ļ����ڸ���ά�ȵ�DOA/AOA����
    %% By Xuliang, 20230417
    %% arrData: ��������
    %% cfgDOA: ��������

    addpath('./DOA_FUNC'); % ����DOA����·��
    
    doaMethod = cfgDOA.EleMethod; % ��ȡdoa���Ʒ��� ÿ�ַ������ؿռ���
    
    if strcmp(doaMethod,'FFT')
        P = cfgDOA.ElesigNum; % P Ϊ��Դ��Ŀ
        [Pout] = DOA_FFT(arrData, cfgDOA);
        [peakVal, peakIdx] = findpeaks(abs(Pout)); % �����ռ��׷�
        [sortVal, sortIdx] = sort(peakVal); % �Է�ֵ��������
        
        ampVal = sortVal(end-P+1:end); % ѡ�����ֵ
        Idx = peakIdx(sortIdx(end)); % ѡ�����ֵ
        
        freqGrids = linspace(-pi, pi, cfgDOA.FFTNum); % Ƶ������
        freq = freqGrids(Idx);
        angleVal = asind(freq / 2 / pi * 2); % �Ƕ�ֵ
        
    elseif strcmp(doaMethod,'MUSIC')
        P = cfgDOA.ElesigNum; % P Ϊ��Դ��Ŀ
        thetaGrids = cfgDOA.thetaGrids; % ���񻮷�
        [Pout] = DOA_MUSIC(arrData, P, thetaGrids); % MUSIC��arrData����ά�ȿ�����M*snap M��Ԫ snap����
        
        [peakVal, peakIdx] = findpeaks(abs(Pout)); % �����ռ��׷�
        [sortVal, sortIdx] = sort(peakVal); % �Է�ֵ��������
        
        ampVal = sortVal(end-P+1:end); % ѡ�����ֵ
        Idx = peakIdx(sortIdx(end-P+1:end)); % ѡ�����ֵ
        angleVal = thetaGrids(Idx); % �Ƕ�ֵ
        
    end
    
    
    doaOut = {};
    doaOut.peakVal = ampVal; % ��ֵ
    doaOut.angleVal = angleVal; % �Ƕ�ֵ
    doaOut.angleIdx = Idx; % �׷�����
    doaOut.spectrum = Pout; % ���ռ���

end