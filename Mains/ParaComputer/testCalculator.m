clc;clear;close all;
% By Xuliang,22134033@zju.edu.cn

Slope = 46.397e12; 
adcNum = 256; 
chirpNum = 128;
fsample = 6847e3;
ramptime = 80e-6; % adc start time ÐÞ¸ÄÎª2us
ideltime = 30e-6;
mode = "complex2x";

[params] = AutoCalculator(Slope, adcNum, chirpNum, fsample, ramptime, ideltime, mode);
