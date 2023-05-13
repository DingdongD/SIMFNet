function [params] = AutoCalculator(Slope, adcNum, chirpNum, fsample, ramptime, ideltime, mode)
    % Slope�� ɨƵб��
    % adcNum�� ��������
    % chirpNum�������ź���Ŀ
    % fsample�� ����Ƶ��
    % ramptime����Ч����ʱ��
    % ideltime�� ����ʱ��
    % mode������ģʽ complex1x complex2x real
    % By Xuliang,22134033@zju.edu.cn
    
    f0 = 77e9; % ����Ƶ��
    txNum = 3; % ����������Ŀ
    TC = (ramptime + ideltime) * txNum; % һ��TDM��chirp����ʱ��
    TF = chirpNum * TC; % ֡��ʱ��

    validB = 1 / fsample * adcNum * Slope; % ��Ч����
    totalB = ramptime * Slope; % �����Ĵ���
    rangeRes = 3e8 / (2 * validB); % ����ֱ���
    maxRange = 3e8 * fsample / (2 * Slope); % ������ 
    if mode == "complex1x"
        maxRange = maxRange;
    else % complex2x��real��Ӧ�����ɼ�������Ҫ����2
        maxRange = maxRange / 2;
    end
    
    velocityRes = 3e8 / (2 *  TF * f0); % �ٶȷֱ���
    maxVelocity = 3e8 / (4 * f0 * TC); % ����ٶ�
    
    params = {};
    params.rangeRes = rangeRes;
    params.maxRange = maxRange;
    params.velocityRes = velocityRes;
    params.maxVelocity = maxVelocity;
    
    disp(["��Ч����(MHZ):",num2str(validB/1e6),"������(MHZ):",num2str(totalB/1e6)]);
    disp(["����ֱ���(m):",num2str(rangeRes),"�ٶȷֱ���(m/s):",num2str(velocityRes)]);
    disp(["������(m):",num2str(maxRange),"����ٶ�(m/s):",num2str(maxVelocity)])
end