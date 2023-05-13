function RadarConfigure
    % By Xuliang,22130433@zju.edu.cn
    addpath(genpath('.\'))
    RSTD_DLL_Path = 'C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Clients\RtttNetClientController\RtttNetClientAPI.dll';
    ErrStatus = Init_RSTD_Connection(RSTD_DLL_Path);
    if (ErrStatus ~= 30000)
        disp('Error inside Init_RSTD_Connection');
        return;
    end
%     strFilename ='C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\Scripts\\DataCaptureDemo_1243_5.lua'; % ��Ӿ��ʵ�����
%     strFilename ='C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\Scripts\\DataCaptureDemo_1243_9.lua'; % ·��ʵ��
    strFilename ='C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\Scripts\\DataCaptureDemo_1243.lua'; % �Ƿ�ʵ��
    Lua_String = sprintf('dofile("%s")',strFilename);
    ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);
end
