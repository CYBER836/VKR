import pandas as pd

file = r'C:\Паша\Проекты\VKR\Monday-WorkingHours.pcap_ISCX.csv'

# читаем ТОЛЬКО первую строку
sample = pd.read_csv(file, nrows=0)   # загрузим только шапку
print('Колонки в файле:')
for i, c in enumerate(sample.columns, 1):
    print(f'{i:02d} | {c}')