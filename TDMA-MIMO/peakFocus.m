function [rd_peak_list, rd_peak] = peakFocus(RDM, cfar_out_list)
    rd_peak = zeros(size(RDM)); % ���ڴ�ŷ�ֵ����
    rd_peak_list = []; % ���ڴ�ŷ�ֵ����
    data_length = size(cfar_out_list, 1); % ���ݳ���
    
    for target_idx = 1 :data_length 
       range_idx = cfar_out_list(target_idx,1);
       doppler_idx = cfar_out_list(target_idx,2);
       
       if range_idx > 1 && range_idx < size(RDM,1) && ...
               doppler_idx > 1 && doppler_idx < size(RDM,2)
           if RDM(range_idx,doppler_idx) > RDM(range_idx-1,doppler_idx) && ...
                   RDM(range_idx, doppler_idx) > RDM(range_idx+1,doppler_idx) && ...
                   RDM(range_idx, doppler_idx) > RDM(range_idx,doppler_idx-1) && ...
                   RDM(range_idx, doppler_idx) > RDM(range_idx,doppler_idx+1) 
               
               rd_peak(range_idx, doppler_idx) = RDM(range_idx,doppler_idx);
               rdlist = [range_idx; doppler_idx];
               rd_peak_list = [rd_peak_list rdlist];         
           end
       end
    end
end