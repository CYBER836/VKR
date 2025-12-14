import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. загружаем и берём 50 000 случайных строк
df = pd.read_csv(r'C:\Паша\Проекты\VKR\cicids2017_clean.csv')
sample = df.sample(n=50_000, random_state=42)

X = sample.drop(columns=['Label'])
y = sample['Label']

# 1.1 чистим inf/NaN
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]

# 1.2 масштабируем
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. t-SNE (2-D)
print('Запускаем t-SNE...')
tsne = TSNE(n_components=2, random_state=42, max_iter=400, verbose=1)
X_tsne = tsne.fit_transform(X_scaled)

# 3. DataFrame для визуализации
tsne_df = pd.DataFrame(data=X_tsne, columns=['t-SNE-1', 't-SNE-2'])
tsne_df['Label'] = y.values

# 4. рисуем
plt.figure(figsize=(8, 6))
sns.scatterplot(data=tsne_df, x='t-SNE-1', y='t-SNE-2', hue='Label',
                alpha=0.7, s=10, legend=False)
plt.title('t-SNE: 16-D → 2-D (50 000 точек)')
plt.tight_layout()
plt.savefig('tsne_2d.png', dpi=300)
plt.show()