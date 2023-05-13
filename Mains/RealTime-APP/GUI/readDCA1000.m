function [VALUE] = readDCA1000(FILENAME)
%% �ļ�Ŀ��: ��ȡ��AWR1243/1443�״�ɼ��� Bin File
%% VALUE: ���ź����ݣ�ά��Ϊ4���������ߣ�*��ADC_SAMPLE * CHIRP_NUM * TX_NUM * FRAME_LENGTH��
%% BY YUXULIANG,2021101

    NUMADCBITS = 16; % ADC��������
    NUMLANES = 4; % ��������ͨ����Ŀ��ͨ������Ҫ�ı䣬���ǽ��õ�ͨ��
    ISREAL = 0; % 0 ��ʾ������1��ʾʵ��

    %% ��ȡ�ļ���תΪ�з�����
    FID = fopen(FILENAME,'r'); % ��ȡbin�ļ�
    ADCDATA = fread(FID, 'int16');

    % ���ADC����Ϊ12λ/14λ����Ҫ�Բ������ݲ���
    if NUMADCBITS ~= 16
        LMAX = 2^(NUMADCBITS-1)-1;
        ADCDATA(ADCDATA > LMAX) = ADCDATA(ADCDATA > LMAX) - 2^NUMADCBITS;
    end
    fclose(FID); % �ر�bin�ļ�

    %% �ο�TI��ADC RAW DATA CAPTURE��PDF ����ͨ������
    if ISREAL
        ADCDATA = reshape(ADCDATA, NUMLANES, []); % ÿ����������ֻ��һ�������ʵ��
    else % ÿ����������������������ݣ�ʵ�����鲿
        ADCDATA = reshape(ADCDATA, NUMLANES*2, []);
        ADCDATA = ADCDATA([1,2,3,4],:) + sqrt(-1)*ADCDATA([5,6,7,8],:);
    end
    VALUE = ADCDATA;
    
end