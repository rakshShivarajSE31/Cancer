import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
#%config InlineBackend.figure_format='retina'
sns.set(style="white")
df = pd.read_excel("bcancer.xlsx")
print(df.head())


df.mitosis = df.mitosis.apply(lambda x: 1 if x == True else 0)
import pandas_summary as ps
summary = ps.DataFrameSummary(df).summary()
summary.loc["missing",:].value_counts()


#subset the non-gene expression variables
#subset = df.loc[:, :"barcode"]

#check distribution for outliers
subset = df.loc[:, "diagnosis"]
subset.hist(figsize=(7,7))
subset = df.loc[:, "mitosis"]
subset.hist(figsize=(7,7))
subset = df.loc[:, "diagnosis"]
subset.hist(figsize=(7,7))
plt.show()