function SendCaptureCMD(data_name)
%% 本文件用于MATLAB发送指令给mmwave 控制DCA采集并回传数据
    root_path = 'H:\RadarProcessing\DataFile\TestData\'; % 根路径名称
    data_path = strcat(root_path,data_name); 
    mkdir(data_path); % 创建文件夹
    
    %% 修改采集数据的脚本文件
    str1 = strcat('adc_data_path="H:\\RadarProcessing\\DataFile\\sea_data\\',data_name,'\\adc_data.bin"'); % 设计bin文件目录
    str = [str1,"ar1.CaptureCardConfig_StartRecord(adc_data_path, 1)","RSTD.Sleep(1000)","ar1.StartFrame()"];
    fid = fopen('C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Scripts\CaptureData1243.lua','w');
    for i = 1:length(str)
        fprintf(fid,'%s\n',str(i));
    end
    fclose(fid); % 关闭文件
    
    %% 配置雷达数据采集
    addpath(genpath('.\'))
    % Initialize mmWaveStudio .NET connection
    RSTD_DLL_Path = 'C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Clients\RtttNetClientController\RtttNetClientAPI.dll';
    ErrStatus = Init_RSTD_Connection(RSTD_DLL_Path);
    if (ErrStatus ~= 30000)
        disp('Error inside Init_RSTD_Connection');
        return;
    end
    strFilename = 'C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\Scripts\\CaptureData1243.lua';
    Lua_String = sprintf('dofile("%s")',strFilename);
    ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);
end
