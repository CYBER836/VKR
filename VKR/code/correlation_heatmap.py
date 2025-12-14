import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Паша\Проекты\VKR\cicids2017_clean.csv')

print(df.columns.tolist())  # ← проверь, как точно называется колонка

plt.figure(figsize=(12, 10))
corr = df.drop(columns=['Label']).corr()  # ← без пробела
sns.heatmap(corr, cmap='coolwarm', square=True, linewidths=.5)
plt.title('Корреляционная матрица признаков')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()