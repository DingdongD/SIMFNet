adc_data_path="H:\\MyDataset\\DatasetFile\\CT10\\adc_data.bin"
ar1.CaptureCardConfig_StartRecord(adc_data_path, 1)
RSTD.Sleep(1000)
ar1.StartFrame()

