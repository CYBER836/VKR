import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'C:\Паша\Проекты\VKR\cicids2017_clean.csv')
X = df.drop(columns=['Label'])

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(X.corr(), dtype=bool))   # нижний треугольник
sns.heatmap(X.corr(), mask=mask, cmap='coolwarm', square=True,
            linewidths=.5, annot_kws={"size": 6}, fmt=".2f",
            annot=True, vmin=-1, vmax=1)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('Корреляционная матрица (lower-triangle)')
plt.tight_layout()
plt.savefig('correlation_heatmap_nice.png', dpi=300)
plt.show()