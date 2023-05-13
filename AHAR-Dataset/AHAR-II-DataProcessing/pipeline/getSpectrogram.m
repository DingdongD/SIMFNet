function [range_time_spectrogram,doppler_time_spectrogram,cadence_velocity_data] = getSpectrogram(adc_data,tx_num,tx_idx,adc_num,chirp_num,range_fft_num,doppler_fft_num,frame_num,overlap)
    % By Xuliang,22134033@zju.edu.cn
    % adc_data : (1,adc_num * tx_num * chirp_num * frame_num)
    % adc_num : default-256
    % tx_num : default-3
    % chirp_num : default-128
    % overlap : default-0.5
    
    range_time_spectrogram = zeros(adc_num / 2,frame_num / overlap);
    doppler_time_spectrogram = zeros(chirp_num,frame_num / overlap);
    cadence_velocity_data = zeros(chirp_num,frame_num / overlap);
    
    rx1_data = reshape(adc_data,adc_num,tx_num,chirp_num,frame_num);
    tx1_data = squeeze(rx1_data(:,tx_idx,:,:));
    siso_data = reshape(tx1_data,adc_num,[]);
    
    for frame_idx = 1 : (frame_num - 1) / overlap + 1
        chosen_data = siso_data(:,[1:chirp_num]+chirp_num*overlap*(frame_idx-1));
        chosen_data = chosen_data - mean(chosen_data,1);     % DCF
        chosen_data = chosen_data .* hanning(adc_num);       % ADD WINDOW TO SUPPRESS SIDELOBES
        range_profile = fft(chosen_data, range_fft_num, 1);  % RANGE FFT
 
        range_profile = range_profile - repmat(mean(range_profile'),size(range_profile,2),1)';  % MTI
        doppler_profile = fftshift(fft(range_profile,doppler_fft_num,2), 2); % DOPPLER FFT
        
        rsum = (sum(doppler_profile(1:end/2,:).^2,2));
        vsum = (sum(doppler_profile(1:end/2,:).^2,1));
        
        doppler_time_spectrogram(:, frame_idx) = vsum;  
        range_time_spectrogram(:, frame_idx) = rsum;  
        cadence_velocity_data(:, frame_idx) = sum(abs(doppler_profile(1:end/2,:)),1);  
    end
    
    doppler_time_spectrogram(:,end - 1 / overlap + 2 : end) = repmat(doppler_time_spectrogram(:,(frame_num - 1) / overlap + 1),1,1 / overlap - 1);
    range_time_spectrogram(:,end - 1 / overlap + 2 : end) = repmat(range_time_spectrogram(:,(frame_num - 1) / overlap + 1),1,1 / overlap - 1);
    cadence_velocity_data(:,end - 1 / overlap + 2 : end) = repmat(cadence_velocity_data(:,(frame_num - 1) / overlap + 1),1,1 / overlap - 1);
    
    ratio = 0.8; % overlap rate, you can modify this to control the output number of radar maps
    doppler_time_spectrogram = imresize(doppler_time_spectrogram, [size(doppler_time_spectrogram,1), size(doppler_time_spectrogram,2)*ratio]);
    range_time_spectrogram = imresize(range_time_spectrogram, [size(range_time_spectrogram,1), size(range_time_spectrogram,2)*ratio]);
    cadence_velocity_data = imresize(cadence_velocity_data, [size(cadence_velocity_data,1), size(cadence_velocity_data,2)*ratio]);
    
end
