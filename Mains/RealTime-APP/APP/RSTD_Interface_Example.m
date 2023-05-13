%% RSTD_Interface_Example.m
addpath(genpath('.\'))

% Initialize mmWaveStudio .NET connection
RSTD_DLL_Path ='C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Clients\RtttNetClientController\RtttNetClientAPI.dll';

global saveName
global framevalue

ErrStatus = Init_RSTD_Connection(RSTD_DLL_Path);
if (ErrStatus ~= 30000)
    fprintf(2,['文件 【',saveName,'】 采集失败\n'])
    error('Error inside Init_RSTD_Connection');
    %return;
end

% %Example Lua Command
% strFilename = 'B:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\Scripts\\DataCaptureDemo_1243_auto.lua';
% Lua_String = sprintf('dofile("%s")',strFilename);
% ErrStatus =RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);

%% 自动改名字 修改lua文件实现重命名
strFilename = 'C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\Scripts\\DataCaptureDemo_1243_auto2.lua';
%saveName='name_test1';
savePath=['D:\\RadarProcessing\\RawData\\',saveName,'.bin'];

fileID=fopen(strFilename,'w');
if ~isempty(framevalue)
    fwrite(fileID,['ar1.FrameConfig(0, 2,',num2str(framevalue),', 255, 80 , 0, 0, 1)']);
    fprintf(fileID,'\n');
end
fwrite(fileID,['ar1.CaptureCardConfig_StartRecord("',savePath,'", 1)']);
fprintf(fileID,'\n');   
fwrite(fileID,['ar1.StartFrame()']);
fclose(fileID);

disp(['文件 【',saveName,'】 开始采集']);

Lua_String = sprintf('dofile("%s")',strFilename);
ErrStatus =RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);

