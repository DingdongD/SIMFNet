function [params] = AutoCalculator(Slope, adcNum, chirpNum, fsample, ramptime, ideltime, mode)
    % Slope： 扫频斜率
    % adcNum： 采样点数
    % chirpNum：单音信号数目
    % fsample： 采样频率
    % ramptime：有效工作时间
    % ideltime： 空闲时间
    % mode：工作模式 complex1x complex2x real
    % By Xuliang,22134033@zju.edu.cn
    
    f0 = 77e9; % 中心频率
    txNum = 3; % 发射天线数目
    TC = (ramptime + ideltime) * txNum; % 一组TDM的chirp持续时间
    TF = chirpNum * TC; % 帧间时间

    validB = 1 / fsample * adcNum * Slope; % 有效带宽
    totalB = ramptime * Slope; % 完整的带宽
    rangeRes = 3e8 / (2 * validB); % 距离分辨率
    maxRange = 3e8 * fsample / (2 * Slope); % 最大距离 
    if mode == "complex1x"
        maxRange = maxRange;
    else % complex2x和real对应的最大可检测距离需要除以2
        maxRange = maxRange / 2;
    end
    
    velocityRes = 3e8 / (2 *  TF * f0); % 速度分辨率
    maxVelocity = 3e8 / (4 * f0 * TC); % 最大速度
    
    params = {};
    params.rangeRes = rangeRes;
    params.maxRange = maxRange;
    params.velocityRes = velocityRes;
    params.maxVelocity = maxVelocity;
    
    disp(["有效带宽(MHZ):",num2str(validB/1e6),"最大带宽(MHZ):",num2str(totalB/1e6)]);
    disp(["距离分辨率(m):",num2str(rangeRes),"速度分辨率(m/s):",num2str(velocityRes)]);
    disp(["最大距离(m):",num2str(maxRange),"最大速度(m/s):",num2str(maxVelocity)])
end