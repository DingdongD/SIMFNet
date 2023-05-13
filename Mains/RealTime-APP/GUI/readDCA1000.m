function [VALUE] = readDCA1000(FILENAME)
%% 文件目的: 读取由AWR1243/1443雷达采集的 Bin File
%% VALUE: 重排后数据，维度为4（接收天线）*（ADC_SAMPLE * CHIRP_NUM * TX_NUM * FRAME_LENGTH）
%% BY YUXULIANG,2021101

    NUMADCBITS = 16; % ADC采样精度
    NUMLANES = 4; % 接收天线通道数目，通常不需要改变，除非仅用单通道
    ISREAL = 0; % 0 表示复数，1表示实数

    %% 读取文件并转为有符号数
    FID = fopen(FILENAME,'r'); % 读取bin文件
    ADCDATA = fread(FID, 'int16');

    % 如果ADC精度为12位/14位则需要对采样数据补偿
    if NUMADCBITS ~= 16
        LMAX = 2^(NUMADCBITS-1)-1;
        ADCDATA(ADCDATA > LMAX) = ADCDATA(ADCDATA > LMAX) - 2^NUMADCBITS;
    end
    fclose(FID); % 关闭bin文件

    %% 参考TI的ADC RAW DATA CAPTURE的PDF 重排通道数据
    if ISREAL
        ADCDATA = reshape(ADCDATA, NUMLANES, []); % 每个接收天线只有一组采样即实部
    else % 每个接收天线有两组采样数据：实部和虚部
        ADCDATA = reshape(ADCDATA, NUMLANES*2, []);
        ADCDATA = ADCDATA([1,2,3,4],:) + sqrt(-1)*ADCDATA([5,6,7,8],:);
    end
    VALUE = ADCDATA;
    
end