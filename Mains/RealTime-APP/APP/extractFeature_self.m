function [feature_self]=extractFeature_self(vf,rf)
%%self 自提特征
    vtmp=10*log10(abs(vf));
    rtmp=10*log10(abs(rf));
    %SVD
    [U,S,V] = svd(vtmp);
    a1=[mean(U(:,1:3)),std(U(:,1:3)),kurtosis(U(:,1:3)),skewness(U(:,1:3))];
    a2=[mean(V(:,1:3)),std(V(:,1:3)),kurtosis(V(:,1:3)),skewness(V(:,1:3))];
    a3=[sum(sum(U)),sum(sum(V)),trace(U),trace(V)];
    %全局
    a4 = entropy(mat2gray(vtmp));
    a5 = [mean(mean(vtmp)),std(std(vtmp)),kurtosis(kurtosis(vtmp)),skewness(skewness(vtmp))];
    a6=[std(mean(vtmp,1)),kurtosis(mean(vtmp,1)),skewness(mean(vtmp,1))...
        std(mean(vtmp,2)),kurtosis(mean(vtmp,2)),skewness(mean(vtmp,2))];
    a=[a1,a2,a3,a4,a5,a6];
    
    %SVD
    [U,S,V] = svd(rtmp);
    b1=[mean(U(:,1:3)),std(U(:,1:3)),kurtosis(U(:,1:3)),skewness(U(:,1:3))];
    b2=[mean(V(:,1:3)),std(V(:,1:3)),kurtosis(V(:,1:3)),skewness(V(:,1:3))];
    b3=[sum(sum(U)),sum(sum(V)),trace(U),trace(V)];
    %全局
    b4 = entropy(mat2gray(rtmp));
    b5 = [mean(mean(rtmp)),std(std(rtmp)),kurtosis(kurtosis(rtmp)),skewness(skewness(rtmp))];
    b6=[std(mean(rtmp,1)),kurtosis(mean(rtmp,1)),skewness(mean(rtmp,1))...
        std(mean(rtmp,2)),kurtosis(mean(rtmp,2)),skewness(mean(rtmp,2))];
    b=[b1,b2,b3,b4,b5,b6];
    
    feature_self=[a,b];
end