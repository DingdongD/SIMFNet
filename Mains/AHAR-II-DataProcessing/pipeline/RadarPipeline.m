function RadarPipeline(adc_data,tx_num,tx_idx,adc_num,chirp_num,range_fft_num,doppler_fft_num,frame_num,overlap,range_of_interest,frame_interval,folder)
    % By Xuliang,22134033@zju.edu.cn
    
    [range_time_spectrogram,doppler_time_spectrogram,cadence_velocity_data] = getSpectrogram(adc_data,tx_num,tx_idx,adc_num,chirp_num,range_fft_num,doppler_fft_num,frame_num,overlap);
    db_rt_plot = db(range_time_spectrogram(range_of_interest,:)+eps)/2;
    db_ud_plot = db(doppler_time_spectrogram+eps)/2;
    
    init_idx = frame_interval(1);
    end_idx = frame_interval(2);
    interval_length = end_idx - init_idx + 1;
    count = 1;
    ratio_frame = size(range_time_spectrogram,2);
    for frame_idx = 1:floor((ratio_frame - end_idx)/interval_length)
        rt_data = db_rt_plot(:,init_idx+interval_length*(frame_idx-1):end_idx+interval_length*(frame_idx-1));
        ud_data = db_ud_plot(:,init_idx+interval_length*(frame_idx-1):end_idx+interval_length*(frame_idx-1));
        cvd_data = cadence_velocity_data(:,init_idx+interval_length*(frame_idx-1):end_idx+interval_length*(frame_idx-1));
        
        cvd_plot = fftshift(fft(cvd_data,32,2),2);
        cvd_data = db(cvd_plot+eps); 
        
        f_tr = figure;
        f_tr.Visible = "off";
       
        imagesc(rt_data);colormap('jet');caxis([80 100]);axis xy;
        set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w');
        set(gcf,'position',[1500,400, 224, 224]);
        set(gca,'looseInset',[0 0 0 0]);
        axis off;
        f = getframe(gcf);
        folder_head = strcat(folder,'\TR\');
        if ~exist(folder_head,'dir')
            mkdir(folder_head);
        end
        imwrite(f.cdata,strcat(folder_head,num2str(count),'.png'));
        
        f_td = figure;
        f_td.Visible = "off";
        imagesc(ud_data);colormap('jet');caxis([80 100]);axis xy;
        set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w');
        set(gcf,'position',[1500,400, 224, 224]);
        set(gca,'looseInset',[0 0 0 0]);
        axis off;
        f = getframe(gcf);
        folder_head = strcat(folder,'\TD\');
        if ~exist(folder_head,'dir')
            mkdir(folder_head);
        end
        imwrite(f.cdata,strcat(folder_head,num2str(count),'.png'));
        
        f_cvd = figure;
        f_cvd.Visible = "off";
        imagesc(cvd_data);colormap('jet');caxis([95 110]);axis xy;
        set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w');
        set(gcf,'position',[1500,400, 224, 224]);
        set(gca,'looseInset',[0 0 0 0]);
        axis off;
        f = getframe(gcf);
        folder_head = strcat(folder,'\CVD\');
        if ~exist(folder_head,'dir')
            mkdir(folder_head);
        end
        imwrite(f.cdata,strcat(folder_head,num2str(count),'.png'));
        
        count = count + 1;
        
    end
end