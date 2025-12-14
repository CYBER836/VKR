import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Паша\Проекты\VKR\cicids2017_clean.csv')

print(df.shape)
print(df['Label'].value_counts())
plt.figure(figsize=(10,5))
sns.countplot(data=df, y='Label', order=df['Label'].value_counts().index,
              palette='viridis')
plt.xscale('log')                  # ← логарифм по X
plt.title('Распределение классов (CICIDS2017, логарифмическая шкала)')
plt.xlabel('Количество (логарифмическая шкала)')
plt.tight_layout()
plt.savefig('class_distribution_log.png', dpi=300)
plt.show()