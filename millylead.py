import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
#%config InlineBackend.figure_format='retina'
sns.set(style="white")
df = pd.read_excel("millyb.xlsx")
print(df.head())



                          #checking for a missing values
df.eventdeath = df.eventdeath.apply(lambda x: 1 if x == True else 0)
df.chemo = df.chemo.apply(lambda x: 1 if x == True else 0)
df.hormonal = df.hormonal.apply(lambda x: 1 if x == True else 0)
df.amputation = df.amputation.apply(lambda x: 1 if x == True else 0)

import pandas_summary as ps
summary = ps.DataFrameSummary(df).summary()
print(summary.loc["missing", :].value_counts())
                          #subset the non-gene expression variables
subset = df.loc[:, :"gene_id_columns"]
                           #check distribution for outliers
subset.hist(figsize=(9, 9))
plt.show()



                           #what the average patient looks like
print("Mean age: " + "%.3f" %np.mean(df["age"]))
print("Mean tumour grade: " + "%.3f" %np.mean(df["grade"]))
print("Mean tumour diameter: " + "%.3f" %np.mean(df["diam"]))
                   #number of patients in survival and death groups
print("___Number of patients in survival and death group and also in therapy's are:___")
print(df.eventdeath.value_counts())
print(df.chemo.value_counts())
print(df.hormonal.value_counts())
print(df.amputation.value_counts())



                 #visualise distribution of continuous clinical variables in both groups of patients
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(8,3), sharey=True)
pal = [sns.color_palette()[0], sns.color_palette()[1]]
sns.boxplot(x="age", y="eventdeath", orient="h", data=subset, ax=ax1, palette=pal)
sns.boxplot(x="diam", y="eventdeath", orient="h", data=subset, ax=ax2, palette=pal)
sns.boxplot(x="posnodes", y="eventdeath", orient="h", data=subset, ax=ax3, palette=pal)
ax1.set_title("Age", size=13)
ax1.invert_yaxis()
ax1.set_ylabel("eventdeath", size=11)
ax1.set_xlabel("Age")
ax2.set_title("Tumour diameter", size=13)
ax2.set_ylabel("")
ax2.set_xlabel("Diameter (mm)")
ax3.set_title("Positive lymph nodes", size=13)
ax3.set_ylabel("")
ax3.set_xlabel("Number of positive nodes")
plt.tight_layout()
#plt.savefig("visuals/EDA_clinical_features.jpeg")
plt.show()


                 #determine the extent to which the predictor and dependent variables fluctuate together
from scipy.stats import pearsonr

clinical_corrs=pd.DataFrame(columns=["Variable", "Correlation coefficient", "P value"])
i=0
for var in ["age", "diam", "posnodes"]:
    corr, p = pearsonr(subset[var], subset["eventdeath"])
    clinical_corrs.loc[i, "Variable"] = var
    clinical_corrs.loc[i, "Correlation coefficient"] = corr
    clinical_corrs.loc[i, "P value"] = p
    i+=1
print(clinical_corrs)



fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10,4), sharey=True)
died = subset[subset["eventdeath"]==1]
survived = subset[subset["eventdeath"]==0]
ax1.hist(survived.survival, alpha=0.8, color=sns.color_palette()[0], label="Survived")
ax1.hist(died.survival, alpha=0.8, color=sns.color_palette()[1], label="Died")
ax1.legend()
ax2.hist(survived.timerecurrence, alpha=0.8, color=sns.color_palette()[0], label="Survived")
ax2.hist(died.timerecurrence, alpha=0.8, color=sns.color_palette()[1], label="Died")
ax2.legend()
ax1.set_xlabel("Years")
ax1.set_ylabel("Number of patients", size=13)
ax1.set_title("Total survival interval in years", size=14)
ax2.set_xlabel("Years")
ax2.set_title("Disease free interval in years", size=14)
#plt.savefig("visuals/EDA_clinical_time_features.jpeg")
plt.show()


time_corrs=pd.DataFrame(columns=["Variable", "Correlation coefficient", "P value"])
i=0
for var in ["survival", "timerecurrence"]:
    corr, p = pearsonr(subset[var], subset["eventdeath"])
    time_corrs.loc[i, "Variable"] = var
    time_corrs.loc[i, "Correlation coefficient"] = corr
    time_corrs.loc[i, "P value"] = p
    i+=1
print(time_corrs)




                     #visualise relationship for categorical clinical variables
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(8,3), sharey=True)
sns.barplot(x="grade", y="eventdeath", data=subset, ax=ax1, color=sns.color_palette()[2])
sns.barplot(x="angioinv", y="eventdeath", data=subset, ax=ax2, color=sns.color_palette()[2])
sns.barplot(x="lymphinfil", y="eventdeath", data=subset, ax=ax3, color=sns.color_palette()[2])
ax1.set_title("Tumour grade", size=14)
ax1.set_xlabel("Class")
ax1.set_ylabel("Proportion of death", size=12)
ax2.set_title("Angioinvasion", size=14)
ax2.set_ylabel("")
ax2.set_xlabel("Class")
ax3.set_title("Lymphocytic infiltration", size=14)
ax3.set_ylabel("")
ax3.set_xlabel("Class")
plt.tight_layout()
#plt.savefig("visuals/EDA_clinical_categorical_features.jpeg")
plt.show()



                            #visualise relationship for categorical treatment variables
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(8,3), sharey=True)
sns.barplot(x="chemo", y="eventdeath", data=subset, ax=ax1, color=sns.color_palette()[3])
sns.barplot(x="hormonal", y="eventdeath", data=subset, ax=ax2, color=sns.color_palette()[3])
sns.barplot(x="amputation", y="eventdeath", data=subset, ax=ax3, color=sns.color_palette()[3])
ax1.set_title("Chemotherapy", size=14)
ax1.set_ylabel("Proportion of death", size=12)
ax1.set_xlabel("")
ax2.set_title("Hormonal therapy", size=14)
ax2.set_xlabel("")
ax2.set_ylabel("")
ax3.set_title("Amputation", size=14)
ax3.set_xlabel("")
ax3.set_ylabel("")
plt.tight_layout()
#plt.savefig("visuals/EDA_clinical_categorical_treatment_features.jpeg")
plt.show()


fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(5,4))
sns.countplot(x="histtype", hue="eventdeath", ax=ax1, data=subset)
ax1.set_title("Patient count by histopathological class", size=14)
ax1.set_xlabel("Class")
ax1.set_ylabel("Count", size=12)
ax1.legend(("Survived", "Died"), loc="upper right")
plt.tight_layout()
#plt.savefig("visuals/EDA_clinical_features_histtype.jpeg")
plt.show()





#dummy the categorical variables and determine the extent to which they correlate
#with the target variable
clinical_var = ["age", "diam", "posnodes", "survival", "timerecurrence", "grade", "angioinv", "lymphinfil", "chemo", "hormonal", "amputation"]
subset2 = df.loc[:,clinical_var]
dummied = pd.get_dummies(subset2, columns=["grade", "angioinv", "lymphinfil"])
                      #create a dataframe of correlations
dummied_corrs = pd.DataFrame(columns=["Variable", "Correlation coefficient", "P value"])
i=0
for var in dummied.loc[:,"chemo":].columns:
    corr, p = pearsonr(dummied[var], subset["eventdeath"])
    dummied_corrs.loc[i, "Variable"] = var
    dummied_corrs.loc[i, "Correlation coefficient"] = corr
    dummied_corrs.loc[i, "P value"] = p
    i = i+1

                    #graph of intravariable relationships
corr = dummied.corr(method="pearson")
sns.set(style="white")
f, ax = plt.subplots(figsize=(13,13))
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax = sns.heatmap(corr, cmap=cmap, mask=mask, annot=True, vmax=.9, center=0, square=True, linewidths=.5, cbar_kws={"shrink":.5})
ax.set_title("Intravariable relationships", size=20)
#plt.savefig("visuals/EDA_heatmap.jpeg")
plt.show()


                       #breakdown of treatments
                      #statistics for the no treatment group and comparison with the baseline
no_treatment = subset[(subset["chemo"]==False) & (subset["amputation"]==False) & (subset["hormonal"]==False)]
print("Number of patients who had no treatment: " + str(np.shape(no_treatment)[0]))
print("Proportion of death in this group: " + ("%.3f" %np.mean(no_treatment["eventdeath"])))
print("Baseline comparison:")
print(df["eventdeath"].value_counts()/df["eventdeath"].count())








