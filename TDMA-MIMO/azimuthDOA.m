function [doaOut] = azimuthDOA(arrData, cfgDOA)
    %% ���ļ����ڷ�λά�ȵ�DOA/AOA����
    %% By Xuliang, 20230417
    %% arrData: �������� ���뷽λά����*1
    %% cfgDOA: ��������

    addpath('./DOA_FUNC'); % ����DOA����·��
    doaMethod = cfgDOA.AziMethod; % ��ȡdoa���Ʒ��� ÿ�ַ������ؿռ���
    doaOut = {};
    if strcmp(doaMethod,'FFT')
        P = cfgDOA.AzisigNum; % P Ϊ��Դ��Ŀ
        [Pout] = DOA_FFT(arrData, cfgDOA);
        if sum(abs(Pout)) == 0
            doaOut.spectrum = Pout; % �ռ���
        else
            [peakVal, peakIdx] = findpeaks(abs(Pout)); % �����ռ��׷�
            [sortVal, sortIdx] = sort(peakVal); % �Է�ֵ�������� �����״��д���3��Ԫ��ȫΪ0 ������������ڷ�ֵ
            ampVal = sortVal(end-P+1:end); % ѡ�����ֵ
            Idx = peakIdx(sortIdx(end)); % ѡ�����ֵ
            
            freqGrids = linspace(-pi, pi, cfgDOA.FFTNum); % Ƶ������
            freq = freqGrids(Idx);
            angleVal = asind(freq / 2 / pi * 2); % �Ƕ�ֵ
            
            doaOut.peakVal = ampVal; % ��ֵ
            doaOut.angleVal = angleVal; % �Ƕ�ֵ
            doaOut.angleIdx = Idx; % �׷�����
            doaOut.spectrum = Pout; % �ռ���
        end
        
    elseif strcmp(doaMethod,'MUSIC')
        P = cfgDOA.AzisigNum; % P Ϊ��Դ��Ŀ
        thetaGrids = cfgDOA.thetaGrids; % ���񻮷�
        
        [Pout] = DOA_MUSIC(arrData, P, thetaGrids); % MUSIC��arrData����ά�ȿ�����M*snap M��Ԫ snap����
        
        if sum(abs(Pout)) == 0
            doaOut.spectrum = Pout; % �ռ���
        else
            [peakVal, peakIdx] = findpeaks(abs(Pout)); % �����ռ��׷�
            [sortVal, sortIdx] = sort(peakVal); % �Է�ֵ��������

            ampVal = sortVal(end-P+1:end); % ѡ�����ֵ
            Idx = peakIdx(sortIdx(end-P+1:end)); % ѡ�����ֵ
            angleVal = thetaGrids(Idx); % �Ƕ�ֵ
            
            doaOut.peakVal = ampVal; % ��ֵ
            doaOut.angleVal = angleVal; % �Ƕ�ֵ
            doaOut.angleIdx = Idx; % �׷�����
            doaOut.spectrum = Pout; % �ռ���
        end
    end
    
end