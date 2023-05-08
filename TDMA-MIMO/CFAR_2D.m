function [cfarOut] = CFAR_2D(RDM, Pfa, TestCells, GuardCells)
    % rd_map����������
    % Pfa���龯����
    % TestCells�����Ե�Ԫ [4 4]
    % GuardCells��������Ԫ [4 4]
    
    % CFAR�����
    detector = phased.CFARDetector2D('TrainingBandSize',TestCells, ...
    'ThresholdFactor','Auto','GuardBandSize',GuardCells, ...
    'ProbabilityFalseAlarm',Pfa,'Method','SOCA','ThresholdOutputPort',true, 'NoisePowerOutputPort',true);

    N_x = size(RDM,1); % ���뵥Ԫ
    N_y = size(RDM,2); % �����յ�Ԫ
    % ��ȡ��ά�������ڵ�size
    Ngc = detector.GuardBandSize(2); % ����������
    Ngr = detector.GuardBandSize(1); % ����������
    Ntc = detector.TrainingBandSize(2); % �ο�������
    Ntr = detector.TrainingBandSize(1); % �ο�������
    cutidx = [];
    colstart = Ntc + Ngc + 1; % �д���
    colend = N_y + ( Ntc + Ngc); % �д�β
    rowstart = Ntr + Ngr + 1; % �д���
    rowend = N_x + ( Ntr + Ngr); % �д�β
    for m = colstart:colend
        for n = rowstart:rowend
            cutidx = [cutidx,[n;m]]; % ������ά��
        end
    end
    
    ncutcells = size(cutidx,2); % ��ȡ���ڱ�������
    
    rd_map_padding = repmat(RDM, 3, 3); % ��RDM����
    chosen_rd_map = rd_map_padding(N_x+1-Ntr-Ngr:2*N_x+Ntr+Ngr,N_y+1-Ntc-Ngc:2*N_y+Ntc+Ngc);
    
    [dets, ~, noise] = detector(chosen_rd_map,cutidx); % ���CFAR���
    
    cfar_out = zeros(size(chosen_rd_map)); % �������
    noise_out = zeros(size(chosen_rd_map)); % ����
    snr_out = zeros(size(chosen_rd_map)); % �����
    for k = 1:size(dets,1)
        if dets(k) == 1
            cfar_out(cutidx(1,k),cutidx(2,k)) = dets(k); 
            noise_out(cutidx(1,k),cutidx(2,k)) = noise(k);
            snr_out(cutidx(1,k),cutidx(2,k)) = chosen_rd_map(cutidx(1,k),cutidx(2,k));
        end
    end
    
    cfarOut = {};
    cfarOut.cfarMap = cfar_out(Ntr+Ngr+1:Ntr+Ngr+N_x,Ntc+Ngc+1:Ntc+Ngc+N_y);
    cfarOut.snrOut = snr_out(Ntr+Ngr+1:Ntr+Ngr+N_x,Ntc+Ngc+1:Ntc+Ngc+N_y);
    cfarOut.noiseOut = noise_out(Ntr+Ngr+1:Ntr+Ngr+N_x,Ntc+Ngc+1:Ntc+Ngc+N_y);
    cfarOut.snrOut = cfarOut.snrOut ./ (eps + cfarOut.noiseOut);
end