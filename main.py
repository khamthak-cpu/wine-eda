import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine(as_frame=True)
df=wine.frame
df['class']=wine.target

print(df.head())
print(df.shape)

df.info()
print(df.isnull().sum())
print(df.duplicated().sum())

print(df.describe().T)

plt.hist(df['alcohol'],bins=20)
plt.show()

num_cols = df.select_dtypes('number').columns.drop(['target','class'])
fig, axes = plt.subplots(3,5, figsize=(18,9))
for ax,col in zip(axes.flat, num_cols):
    df[col].hist(ax=ax,bins=20, color='steelblue', edgecolor='white')
    ax.set_title(col, fontsize=9)
plt.tight_layout()
plt.show()

sns.boxplot(x='class',y='flavanoids',data=df)
plt.show()

sns.scatterplot(x='flavanoids',y='total_phenols', hue='class',data=df)
plt.show()

corr=df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.1f', cmap='coolwarm',center=0)
plt.show()

print(df.groupby('class')[num_cols].mean().T)

#===== FINDINGS ======
#1. The dataset is clean: 0 missing values in all 15 columns(isnull().sum()), 0 duplicate rows, all numeric types
#2  "Flavanoids best separate the wine classes: class 0 averages 2.98, class 2 only 0.78, with almost no overlap in the boxplot."
# 3. total_phenols and flavanoids are strongly correlated at ~0.9, meaning ...
# 4. proline's scale is much larger than other features,
#which matters because the x_axis range is large compared to other graph
