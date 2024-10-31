import cfgrib
import xarray
import pandas as pd
import numpy as np

# 打开GRIB文件
grib_file = 'data.grib'
ds = xarray.open_dataset(grib_file, engine='cfgrib')
df = ds.to_dataframe().reset_index()  # 对于grib格式，能够直接转换成dataframe
# if 'time' in df.columns:
#     df['time'] = pd.to_datetime(df['time'])    # 如果不是datetime格式则需要转换
df.to_csv('full_dataset.csv', index=False)