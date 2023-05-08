function [PoutMusic] = DOA_MUSIC(X, P, thetaGrids)
    % X: �����ź� Channel * ChirpNum
    % P: Ŀ����Ŀ
    % PoutMusic: ���������
    
    M = size(X, 1); % ��Ԫ��
    snap = size(X, 2); % ������
    RX = X * X' / snap; % Э�������
    
    [V, D] = eig(RX); % ����ֵ�ֽ�
    eig_value = real(diag(D)); % ��ȡ����ֵ
    [B, I] = sort(eig_value, 'descend'); % ��������ֵ
    EN = V(:, I(P+1:end)); % ��ȡ�����ӿռ�
    
    PoutMusic = zeros(1, length(thetaGrids));
    
    for id = 1 : length(thetaGrids)
        atheta_vec = exp(1j * 2 * pi * [0:M-1]' * 1 / 2 * sind(thetaGrids(id))); % ����ʸ��
        PoutMusic(id) = ((1 / (atheta_vec' * EN * EN' * atheta_vec))) ; % �����׼���
    end
end

    
    
   