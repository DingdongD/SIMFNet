function [VALUE] = readDCA1000(FILENAME)
%% FUNC: Read Bin File collected by AWR1243/1443 radar
%% VALUE: Rearranged data,£¨TX_NUM£©*£¨ADC_SAMPLE * CHIRP_NUM * TX_NUM * FRAME_LENGTH£©
%% BY XULIANG,22134033@zju.edu.cn

    NUMADCBITS = 16; % ADC sampling accuracy
    NUMLANES = 4; % TX_NUM, which usually does not need to be changed, unless only a single channel is used
    ISREAL = 0; % 0 for complex and 1 for real mode

    %% read files and convert to sign numbers
    FID = fopen(FILENAME,'r'); % read bin file
    ADCDATA = fread(FID,'int16');
%     ADCDATA = fread(FID,37498880,'int16');
    % If the ADC accuracy is 12/14 bits then the sampled data needs to be compensated
%     disp(size(ADCDATA,1));
    if NUMADCBITS ~= 16
        LMAX = 2^(NUMADCBITS-1)-1;
        ADCDATA(ADCDATA > LMAX) = ADCDATA(ADCDATA > LMAX) - 2^NUMADCBITS;
    end
    fclose(FID); % close bin file 

    %% Refer to TI's ADC RAW DATA CAPTURE PDF to rearrange channel data
    if ISREAL % Each receiving antenna has only one set of samples, i.e., the real part
        ADCDATA = reshape(ADCDATA, NUMLANES, []); 
    else % Each receive antenna has two sets of sampled data: the real part and the image part
        ADCDATA = reshape(ADCDATA, NUMLANES*2, []);
        ADCDATA = ADCDATA([1,2,3,4],:) + sqrt(-1)*ADCDATA([5,6,7,8],:);
    end
    
    VALUE = ADCDATA(1,:);
%     VALUE = ADCDATA;

end