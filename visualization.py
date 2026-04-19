import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

print("Correlation Matrix:")
print(df.corr())

sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()