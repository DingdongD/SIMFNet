function [md_feature,cfs_feature,rt_feature] = extract_feature(velocity_axis,db_ud_plot,db_rt_plot,db_cvd_plot,total_num)

    %% kurtosis,skewness,mean,standard deviation and peak of  torso_frequency in TDM
    torso_frequency = zeros(1,total_num);
    for idx = 1 : total_num
        frac_son = sum(velocity_axis(1,db_ud_plot(:,1) > 0)' .* db_ud_plot(db_ud_plot(:,1) > 0,idx));
        frac_mother = sum(db_ud_plot(db_ud_plot(:,1) > 0,idx));
        torso_frequency(idx) = frac_son / frac_mother;
    end
    mean_torso = mean(torso_frequency);
    std_torso = std(torso_frequency);
    skewness_torso = skewness(torso_frequency);
    kurtosis_torso = kurtosis(torso_frequency);
    
    %% kurtosis,skewness,mean,standard deviation and peak of  limb_frequency in TDM
    limb_frequency = zeros(1,total_num);
    for idx = 1 : total_num
        frac_son = sum((velocity_axis(1,db_ud_plot(:,1) > 0)' .* db_ud_plot(db_ud_plot(:,1) > 0,idx) -...
            torso_frequency(idx) * db_ud_plot(db_ud_plot(:,1) > 0,idx)) .^ 2);
        frac_mother = sum(db_ud_plot(db_ud_plot(:,1) > 0,idx));
        limb_frequency(idx) = sqrt(frac_son / frac_mother);
    end
    mean_limb = mean(limb_frequency);
    std_limb = std(limb_frequency);
    skewness_limb = skewness(limb_frequency);
    kurtosis_limb = kurtosis(limb_frequency);

    %% kurtosis,skewness,mean,standard deviation and peak of TDM
    ud_2d_skewness = skewness(skewness(db_ud_plot));
    ud_2d_kurtosis = skewness(kurtosis(db_ud_plot));
    ud_2d_mean = mean(mean(db_ud_plot));
    ud_2d_std = std(std((db_ud_plot)));
    ud_2d_entropy = entropy(mat2gray(db_ud_plot));

    %% SVD decomposition of TDM
    [U,S,V] = svd(db_ud_plot);
    % Select the energy and mean values of the left and right singular matrices
    sum_u = sum(sum(U));
    sum_v = sum(sum(V));
    mean_eig_u = mean(U(:,1));
    mean_eig_v = mean(V(:,1));
    mean_diag = mean(diag(S));
    
    md_feature = [mean_torso,std_torso,skewness_torso,kurtosis_torso,mean_limb,std_limb,skewness_limb,kurtosis_limb,ud_2d_skewness,ud_2d_kurtosis,ud_2d_mean,ud_2d_std,ud_2d_entropy,sum_u,sum_v,mean_eig_u,mean_eig_v,mean_diag];
    
    %% kurtosis,skewness,mean,standard deviation and peak of CVD
    db_cfs_plot = sum(db_cvd_plot,1);
    cfs_1d_skewness = skewness(db_cfs_plot);
    cfs_1d_kurtosis = kurtosis(db_cfs_plot);
    cfs_1d_mean = mean(db_cfs_plot);
    cfs_1d_std = std(db_cfs_plot);
    cfs_1d_peak = max(db_cfs_plot);
    cfs_feature = [cfs_1d_skewness,cfs_1d_kurtosis,cfs_1d_mean,cfs_1d_std,cfs_1d_peak];
    
    %% kurtosis,skewness,mean and standard deviation of TRM
    rt_2d_skewness = skewness(skewness(db_rt_plot));
    rt_2d_kurtosis = skewness(kurtosis(db_rt_plot));
    rt_2d_mean = mean(mean(db_rt_plot));
    rt_2d_std = std(std((db_rt_plot)));
    rt_2d_entropy = entropy(mat2gray(db_rt_plot));
    rt_feature = [rt_2d_skewness,rt_2d_kurtosis,rt_2d_mean,rt_2d_std,rt_2d_entropy]; 
end