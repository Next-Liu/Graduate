import pandas as pd
import os

folderList = ['../data/20200101-20201231', '../data/20210101-20211231', '../data/20220101-20221231',
              '../data/20230101-20231231']
folderPath = '../Data/20200101-20201231'
#folderPath = '../data/test'

final_data = []

# 遍历df
# for fileName in os.listdir(folderPath):
#     df = pd.read_csv(folderPath + '/' + fileName)
#     temp_hour = 0
#     count = 0
#     temp_data = []
#     print(fileName)
#     for i in range(len(df)):
#         date = df.iloc[i]['date']
#         hour = df.iloc[i]['hour']
#         if hour == 0 and count == 0:
#             temp_data = [date, hour]
#         if hour == temp_hour:
#             count += 1
#             if df.iloc[i]['type'] == 'AQI':
#                 temp_data.append(df.iloc[i]['1036A'])
#             if df.iloc[i]['type'] == 'PM2.5':
#                 value = df.iloc[i]['1036A']
#                 temp_data.append(value if value != '' else 0)
#             if df.iloc[i]['type'] == 'PM10':
#                 value = df.iloc[i]['1036A']
#                 temp_data.append(value if value != '' else 0)
#             if df.iloc[i]['type'] == 'SO2':
#                 value = df.iloc[i]['1036A']
#                 temp_data.append(value if value != '' else 0)
#             if df.iloc[i]['type'] == 'NO2':
#                 value = df.iloc[i]['1036A']
#                 temp_data.append(value if value != '' else 0)
#             if df.iloc[i]['type'] == 'O3':
#                 value = df.iloc[i]['1036A']
#                 temp_data.append(value if value != '' else 0)
#             if df.iloc[i]['type'] == 'CO':
#                 value = df.iloc[i]['1036A']
#                 temp_data.append(value if value != '' else 0)
#         if hour != temp_hour:
#             final_data.append(temp_data)
#             temp_hour += 1
#             temp_data = [date, temp_hour]
#             if df.iloc[i]['type'] == 'AQI':
#                 temp_data.append(df.iloc[i]['1036A'])
#         if i == len(df) - 1:
#             final_data.append(temp_data)
# 处理1040A,1041A数据
final_1040_data = []
final_1041_data = []

# 遍历df
for fileName in os.listdir(folderPath):
    df = pd.read_csv(folderPath + '/' + fileName)
    temp_1040_data = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
                      13: [],
                      14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: []}
    temp_1041_data = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
                      13: [],
                      14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: []}
    # 对temp_data每个数组添加上日期
    date = fileName.split('_')[2].split('.')[0]
    print(fileName)
    for i in range(24):
        temp_1040_data.get(i).append(date)
        temp_1040_data.get(i).append(i)
        temp_1041_data.get(i).append(date)
        temp_1041_data.get(i).append(i)
    for i in range(len(df)):
        date = df.iloc[i]['date']
        hour = df.iloc[i]['hour']
        if df.iloc[i]['type'] == 'AQI':
            temp_1040_data.get(hour).append(0 if df.iloc[i]['1040A'] is None else df.iloc[i]['1040A'])
            temp_1041_data.get(hour).append(0 if df.iloc[i]['1041A'] is None else df.iloc[i]['1041A'])
        if df.iloc[i]['type'] == 'PM2.5':
            temp_1040_data.get(hour).append(0 if df.iloc[i]['1040A'] is None else df.iloc[i]['1040A'])
            temp_1041_data.get(hour).append(0 if df.iloc[i]['1041A'] is None else df.iloc[i]['1041A'])
        if df.iloc[i]['type'] == 'PM10':
            temp_1040_data.get(hour).append(0 if df.iloc[i]['1040A'] is None else df.iloc[i]['1040A'])
            temp_1041_data.get(hour).append(0 if df.iloc[i]['1041A'] is None else df.iloc[i]['1041A'])
        if df.iloc[i]['type'] == 'SO2':
            temp_1040_data.get(hour).append(0 if df.iloc[i]['1040A'] is None else df.iloc[i]['1040A'])
            temp_1041_data.get(hour).append(0 if df.iloc[i]['1041A'] is None else df.iloc[i]['1041A'])
        if df.iloc[i]['type'] == 'NO2':
            temp_1040_data.get(hour).append(0 if df.iloc[i]['1040A'] is None else df.iloc[i]['1040A'])
            temp_1041_data.get(hour).append(0 if df.iloc[i]['1041A'] is None else df.iloc[i]['1041A'])
        if df.iloc[i]['type'] == 'O3':
            temp_1040_data.get(hour).append(0 if df.iloc[i]['1040A'] is None else df.iloc[i]['1040A'])
            temp_1041_data.get(hour).append(0 if df.iloc[i]['1041A'] is None else df.iloc[i]['1041A'])
        if df.iloc[i]['type'] == 'CO':
            temp_1040_data.get(hour).append(0 if df.iloc[i]['1040A'] is None else df.iloc[i]['1040A'])
            temp_1041_data.get(hour).append(0 if df.iloc[i]['1041A'] is None else df.iloc[i]['1041A'])
        if i == len(df) - 1:
            for j in range(24):
                if len(temp_1040_data.get(j)) == 9:
                    final_1040_data.append(temp_1040_data.get(j))
                if len(temp_1041_data.get(j)) == 9:
                    final_1041_data.append(temp_1041_data.get(j))

columns = ['date', 'hour', 'AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
# 将final_data的元素依次添加到dataframe中
new_df = pd.DataFrame(columns=columns)
for i in range(len(final_data)):
    new_df = new_df.append(pd.DataFrame([final_data[i]], columns=columns))
new_df.to_csv('1036A_2020.csv', index=False)
