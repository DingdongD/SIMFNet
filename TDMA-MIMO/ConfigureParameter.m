function [cfgOut] = ConfigureParameter()
    %% ���ļ����ڷ������ú��ײ��״�ϵͳ����
    %% By Xuliang, 20230411
    cfgOut.Mode = 1; % ����ģʽ
    cfgOut.ADCNum = 256; % ADC������Ŀ
    cfgOut.ChirpNum = 128; % ÿ֡����Chirp��Ŀ
    cfgOut.Frame = 1; % ��֡��
    cfgOut.applyVmaxExtend = 1; % �����ٶ���չ
    if cfgOut.applyVmaxExtend
        cfgOut.min_dis_apply_vmax_extend = 10; % �����ٶ���չ����С����
    end
    
    if cfgOut.Mode == 1 
        disp(strcat(['=====','��ѡ��AWR1243-�����״�ģʽ','====='])); % �����״�ģʽ3T4R
        cfgOut.numTx = 3;
        cfgOut.numRx = 4;  
        
        cfgOut.PosTX_X = [0 2 4];
        cfgOut.PosTX_Y = [0 1 0]; 
        cfgOut.PosRX_X = [0 1 2 3];
        cfgOut.PosRX_Y = [0 0 0 0];
        cfgOut.PosTX_BOARD_ID = [1 2 3]; % �������߰���˳�� ����û�õ�
        cfgOut.PosTX_Trans_ID = [1 3 2]; % �������߷���˳�� 
        cfgOut.PosRX_X = [0 1 2 3];
        cfgOut.PosRX_Y = zeros(1,cfgOut.numRx); 
        cfgOut.PosRX_BOARD_ID = [1 2 3 4]; % �������߰���˳��
        
    elseif cfgOut.Mode == 2
        disp(strcat(['=====','��ѡ��AWR2243-�����״�ģʽ','====='])); % �����״�ģʽ12T16R
        cfgOut.numTx = 12;
        cfgOut.numRx = 16;
        cfgOut.PosTX_X = [11 10 9 32 28 24 20 16 12 8 4 0];
        cfgOut.PosTX_Y = [6 4 1 0 0 0 0 0 0 0 0 0]; 
        cfgOut.PosTX_BOARD_ID = [12 11 10 3 2 1 9 8 7 6 5 4]; % �������߰���˳�� ����û�õ�
        cfgOut.PosTX_Trans_ID = [12:-1:1]; % �������߷���˳��
        
        cfgOut.PosRX_X = [11 12 13 14 50 51 52 53 46 47 48 49 0 1 2 3];
        cfgOut.PosRX_Y = zeros(1,cfgOut.numRx); 
        cfgOut.PosRX_BOARD_ID = [13 14 15 16 1 2 3 4 9 10 11 12 5 6 7 8]; % �������߰���˳��
    elseif cfgOut.Mode == 3
        disp(strcat(['=====','��ѡ����˹��6T8R-�����״�ģʽ','=====']));
        cfgOut.numTx = 6;
        cfgOut.numRx = 8;
        cfgOut.PosTX_X = [0 17 34 41 48 55];
        cfgOut.PosTX_Y = [0 0 0 0 4 8 12]; 
        cfgOut.PosTX_BOARD_ID = [1 2 3 4 5 6]; % �������߰���˳�� ����û�õ�
        cfgOut.PosTX_Trans_ID = [1 2 3 4 5 6]; % �������߷���˳��
        
        cfgOut.PosRX_X = [0 2 3 5 8 11 14 17];
        cfgOut.PosRX_Y = zeros(1,cfgOut.numRx); 
        cfgOut.PosRX_BOARD_ID = [1 2 3 4 5 6 7 8]; % �������߰���˳��
    end
    
    virtual_azi_arr = repmat(cfgOut.PosTX_X(cfgOut.PosTX_Trans_ID), cfgOut.numRx, 1) + repmat(cfgOut.PosRX_X(cfgOut.PosRX_BOARD_ID), cfgOut.numTx, 1).';
    virtual_ele_arr = repmat(cfgOut.PosTX_Y(cfgOut.PosTX_Trans_ID), cfgOut.numRx, 1) + repmat(cfgOut.PosRX_Y(cfgOut.PosRX_BOARD_ID), cfgOut.numTx, 1).';
    
    virtual_array.azi_arr = reshape(virtual_azi_arr,[],1);
    virtual_array.ele_arr = reshape(virtual_ele_arr,[],1);
    virtual_array.tx_id = reshape(repmat(cfgOut.PosTX_Trans_ID, cfgOut.numRx, 1),[],1);
    virtual_array.rx_id = reshape(repmat(cfgOut.PosRX_BOARD_ID, cfgOut.numTx, 1).',[],1);
    virtual_array.virtual_arr = cat(2,virtual_array.azi_arr,virtual_array.ele_arr,virtual_array.rx_id,virtual_array.tx_id);
    
   % ��ȡ��������Ԫ��
    [~, noredundant_idx] = unique(virtual_array.virtual_arr(:,1:2),'rows'); % ��Ԫ��ȥ��
    virtual_array.noredundant_arr = virtual_array.virtual_arr(noredundant_idx,:); % ��ȡ���������еķ�λ���������շ�ID
    virtual_array.noredundant_aziarr = virtual_array.noredundant_arr(virtual_array.noredundant_arr(:,2)==0,:); % ��ȡ��λά������
    
    % ��ȡ������Ԫ��
    redundant_idx = setxor([1:cfgOut.numRx*cfgOut.numTx],noredundant_idx); % 58 * 1 ���ز��� A �� B �Ľ����е����ݣ��ԳƲ�����������ظ��
    virtual_array.redundant_arr = virtual_array.virtual_arr(redundant_idx,:); % ��ȡ�������еķ�λ���������շ�ID
    
    % ���Ҳ������ص���TX-RX��
    if ~isempty(redundant_idx) % ���岻����������Ԫ
        info_overlaped_associate_arr = [];
        for re_arr_id = 1 : size(virtual_array.redundant_arr,1)
            % �����Ƕ�������Ԫ�ͷ�������Ԫ����ƥ�� ������Ϊ[[T,F,F,T],...]����ʽ ά��Ϊ134*4 ��������
            mask = virtual_array.noredundant_arr == virtual_array.redundant_arr(re_arr_id, :); 
            mask = mask(:,1) & mask(:,2); % �����Ƕ�������Ԫ�ͷ�������Ԫ����ƥ�� ������Ϊ[[T,F,F,T],...]����ʽ ά��Ϊ134*4
            info_associate = virtual_array.noredundant_arr(mask,:); % �ҵ�����ͷ������н��պͷ��������ص���Ԫ��
            info_overlaped = virtual_array.redundant_arr(re_arr_id,:); % ��ǰ������Ԫֵ
            % ���й����ķ�������Ԫ��������Ԫ�����б� �ص�����˼ָ�����ɲ�ͬTX-RX���߶��γɵ���λ������ͬ��
            info_overlaped_associate_arr = [info_overlaped_associate_arr; [info_associate, info_overlaped]];
        end
        diff_tx = abs(info_overlaped_associate_arr(:,8) - info_overlaped_associate_arr(:,4)); % ���㷢�����ߵ�λ�ò�
        info_overlaped_diff1tx_arr = info_overlaped_associate_arr(diff_tx==1,:); % 32*8 �������߲�Ϊ1���ص���Ԫ
        [sotedVal, sortedIdx] = sort(info_overlaped_diff1tx_arr(:, 1)); % ����λ��λ������
        virtual_array.info_overlaped_diff1tx = info_overlaped_diff1tx_arr(sortedIdx, :);
    else
        cfgOut.applyVmaxExtend = 0; % �����ٶ���չ
    end
    
    sig_space_idx0 = virtual_array.noredundant_arr(:,1)+1;
    sig_space_idx1 = virtual_array.noredundant_arr(:,2)+1;
    sig_idx0 = [];
    sig_idx1 = [];
    rx_pos_set = virtual_array.noredundant_arr(:,3); % ��������λ�ü���
    tx_pos_set = virtual_array.noredundant_arr(:,4); % ��������λ�ü���
    for rx_id = 1 :length(virtual_array.noredundant_arr(rx_pos_set))
        rx_arr = rx_pos_set(rx_id);
        sig_idx0 = [sig_idx0, find(cfgOut.PosRX_BOARD_ID == rx_arr)]; % ȷ���������ߵ����
    end
    for tx_id = 1 :length(virtual_array.noredundant_arr(tx_pos_set))
        tx_arr = tx_pos_set(tx_id);
        sig_idx1 = [sig_idx1, find(cfgOut.PosTX_Trans_ID == tx_arr)]; % ȷ���������ߵ�˳����� ���ǰ���������
    end
    cfgOut.sigSpaceIdx = [sig_space_idx0.'; sig_space_idx1.'];
    cfgOut.sigIdx = [sig_idx0; sig_idx1];
    
    cfgOut.virtual_array = virtual_array; % ��virtual_array�����������
    
    cfgOut.arrdx = 0.5; % ��һ����Ԫ���
    cfgOut.arrdy = 0.5; 
    
    % �������δ���Ƿ���˳��ǰ�İ汾
%     arrX = [];
%     arrY = [];
%     for id = 1 : cfgOut.numTx
%        arrX = [arrX PosRX_X + PosTX_X(id)]; % MIMO��Ԫ����
%        arrY = [arrY PosRX_Y + PosTX_Y(id)];
%     end   
%     cfgOut.array(1,:) = arrX;
%     cfgOut.array(2,:) = arrY;
    
    cfgOut.antennaPhase = zeros(1,cfgOut.numRx*cfgOut.numTx); % ��ʼ����λ[���Խ�Ϸ������У׼]
    
    
    % �źŲ���
    cfgOut.startFreq = 76.5e9; % ��ʼƵ��
    cfgOut.fs = 6874e3; % ADC����Ƶ�� Hz 6874e3
    cfgOut.Ramptime = 80e-6; % ����ʱ�� 80e-6
    cfgOut.Idletime = 30e-6; % ����ʱ�� 30e-6
    cfgOut.Slope = 46.397e12; % Chirpб�� 46.397
    cfgOut.validB = 1 / cfgOut.fs * cfgOut.ADCNum * cfgOut.Slope; % ʵ����Ч����
    cfgOut.totalB = cfgOut.Ramptime * cfgOut.Slope; % ��������
    cfgOut.Tc = cfgOut.Idletime + cfgOut.Ramptime; % ��Chirp����
    cfgOut.ValidTc = 1 / cfgOut.fs * cfgOut.ADCNum; % ��Chirp��ADC��Ч����ʱ��
    cfgOut.fc = cfgOut.totalB / 2 + cfgOut.startFreq + 1 / cfgOut.fs * cfgOut.ADCNum; % ��Ƶ Hz ��������Ϊ��Ч����ʼƵ�� 
    
    % Ӳ������
    cfgOut.Pt = 12; % dbm ���书��
    cfgOut.Fn = 12; % ����ϵ��
    cfgOut.Ls = 3;  % ϵͳ���
    
%     cfgOut.Pa = 48; % dB ������·����
    
end