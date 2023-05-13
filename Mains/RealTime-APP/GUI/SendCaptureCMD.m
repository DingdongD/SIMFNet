function SendCaptureCMD(data_name)
%% ���ļ�����MATLAB����ָ���mmwave ����DCA�ɼ����ش�����
    root_path = 'H:\RadarProcessing\DataFile\TestData\'; % ��·������
    data_path = strcat(root_path,data_name); 
    mkdir(data_path); % �����ļ���
    
    %% �޸Ĳɼ����ݵĽű��ļ�
    str1 = strcat('adc_data_path="H:\\RadarProcessing\\DataFile\\sea_data\\',data_name,'\\adc_data.bin"'); % ���bin�ļ�Ŀ¼
    str = [str1,"ar1.CaptureCardConfig_StartRecord(adc_data_path, 1)","RSTD.Sleep(1000)","ar1.StartFrame()"];
    fid = fopen('C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Scripts\CaptureData1243.lua','w');
    for i = 1:length(str)
        fprintf(fid,'%s\n',str(i));
    end
    fclose(fid); % �ر��ļ�
    
    %% �����״����ݲɼ�
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
