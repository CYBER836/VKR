# smote_split.py  (полный листинг)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# 1. загружаем полный набор
df = pd.read_csv(r'C:\\Паша\\Проекты\\VKR\\cicids2017_clean.csv')

# 2. X, y
X = df.drop(columns=['Label'])
y = df['Label']

# 3. чистим inf/NaN
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]

# 4. стратифицированное разбиение 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

print('До SMOTE (train):')
print(y_train.value_counts())

# 5. SMOTE: доводим ВСЕ minority до 80 000 (≈ 1:1 к BENIGN)
smote = SMOTE(random_state=42, sampling_strategy='all', k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print('\nПосле SMOTE (train):')
print(pd.Series(y_train_sm).value_counts())

# 6. сохраняем на диск
np.save(r'C:\\Паша\\Проекты\\VKR\\X_train_smote.npy', X_train_sm)
np.save(r'C:\\Паша\\Проекты\\VKR\\y_train_smote.npy', y_train_sm)
np.save(r'C:\\Паша\\Проекты\\VKR\\X_test.npy', X_test)
np.save(r'C:\\Паша\\Проекты\\VKR\\y_test.npy', y_test)

# 7. график «до/после»
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(y_train, ax=ax[0])
ax[0].set_title('До SMOTE')
sns.countplot(pd.Series(y_train_sm), ax=ax[1])
ax[1].set_title('После SMOTE')
plt.tight_layout()
plt.savefig('smote_balance.png', dpi=300)
plt.show()