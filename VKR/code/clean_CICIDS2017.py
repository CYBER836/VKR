import pandas as pd
import glob
import os

# ---------- настройки ----------
WANTED = [  # колонки, которые мы точно хотим сохранить
    'Destination Port', 'Flow Duration',
    'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'Average Packet Size',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'Label'
]

IN_DIR = r'C:\Паша\Проекты\VKR'
OUT_DIR = os.path.join(IN_DIR, 'clean_parts')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- находим файлы ----------
files = glob.glob(os.path.join(IN_DIR, '*pcap_ISCX.csv'))
print(f'Найдено файлов: {len(files)}')
if not files:
    print('Файлы *pcap_ISCX.csv не найдены!')
    exit()

# ---------- обработка ----------
for file in files:
    fname = os.path.basename(file)
    out_name = fname.replace('.pcap_ISCX.csv', '_clean.csv')
    out_path = os.path.join(OUT_DIR, out_name)

    print(f'➜ {fname}')
    chunk_list = []
    for chunk in pd.read_csv(file, chunksize=100_000):
        # удаляем пробелы в названиях колонок
        chunk.columns = chunk.columns.str.strip()
        # оставляем только нужные, если они есть
        avail = [c for c in WANTED if c in chunk.columns]
        chunk = chunk[avail]
        chunk = chunk.drop_duplicates().dropna()
        chunk_list.append(chunk)

    if not chunk_list:
        print('  ⚠ после фильтрации не осталось строк – пропускаем')
        continue

    df_day = pd.concat(chunk_list, ignore_index=True)
    df_day.to_csv(out_path, index=False)
    print(f'  ✓ сохранено {len(df_day)} строк, колонки: {list(df_day.columns)}')

# ---------- объединение ----------
all_clean = [pd.read_csv(f) for f in glob.glob(os.path.join(OUT_DIR, '*_clean.csv'))]
if not all_clean:
    print('⚠ Нет ни одного clean-файла – объединять нечего')
else:
    full = pd.concat(all_clean, ignore_index=True)
    full.to_csv(os.path.join(IN_DIR, 'cicids2017_clean.csv'), index=False)
    print('\nИтоговый датасет:', full.shape)
    print(full['Label'].value_counts())