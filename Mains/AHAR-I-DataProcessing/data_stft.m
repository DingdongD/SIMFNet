function [rt_plot,ud_plot] = data_stft(adc_data,adc_num,tx_num,tx_idx,chirp_num,frame_num,overlap,min_range,max_range,range_axis)
    % adc_data : (1,adc_num * tx_num * chirp_num * frame_num)
    % adc_num : default-256
    % tx_num : default-3
    % chirp_num : default-128
    % overlap : default-0.5
    
    rt_plot = zeros(adc_num,frame_num / overlap);
    ud_plot = zeros(chirp_num,frame_num / overlap);
    rx1_data = reshape(adc_data,adc_num,tx_num,chirp_num,frame_num);
    tx1_data = squeeze(rx1_data(:,tx_idx,:,:));
    temp_data = reshape(tx1_data,adc_num,[]);
    temp_index = [min_range : max_range];
    for frame_idx = 1 : (frame_num - 1) / overlap + 1
        select_data = temp_data(:,[1:chirp_num]+chirp_num*overlap*(frame_idx-1));
        select_data = select_data - mean(select_data,1); % 滤除静态杂波
        select_data = select_data .* hanning(adc_num); % 加窗
        range_profile = fft(select_data,adc_num,1); % 距离fft 
        range_profile = range_profile - repmat(mean(range_profile'),size(range_profile,2),1)'; % 滤除速度为0目标
        
        doppler_profile = fftshift(fft(range_profile,chirp_num,2),2); % 多普勒fft
        
%         doppler_profile = doppler_profile(1:end/2,:);
        
%         doppler_profile = repmat(range_axis',1,adc_num/2).^2 * doppler_profile;

        dsum = abs(doppler_profile).^2;
        rsum = zeros(size(dsum,1),1);
        rsum(temp_index) = sum(dsum(temp_index,:),2);
        doppler_profile = (abs(doppler_profile(temp_index,:)).^2 );
        ud_plot(:,frame_idx) = sum(doppler_profile).';
        rt_plot(:,frame_idx) = rsum;   
    end
    ud_plot(:,end - 1 / overlap + 2 : end) = repmat(ud_plot(:,(frame_num - 1) / overlap + 1),1,1 / overlap - 1);
    rt_plot(:,end - 1 / overlap + 2 : end) = repmat(rt_plot(:,(frame_num - 1) / overlap + 1),1,1 / overlap - 1);
end
