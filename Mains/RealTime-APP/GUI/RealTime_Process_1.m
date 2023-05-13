%% 实时处理系统设计1
%% 雷达参数配置
addpath(genpath('.\'))
RSTD_DLL_Path = 'C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Clients\RtttNetClientController\RtttNetClientAPI.dll';
ErrStatus = Init_RSTD_Connection(RSTD_DLL_Path);
if (ErrStatus ~= 30000)
    disp('Error inside Init_RSTD_Connection');
    return;
end
strFilename ='C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\Scripts\\DataCaptureDemo_1243_3.lua';
Lua_String = sprintf('dofile("%s")',strFilename);
ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);
