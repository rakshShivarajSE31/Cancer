import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

#data file
data = pd.read_excel("millyb.xlsx")
#feature's array
feature_coles = ['id',	'clump_thickness',	'unif_cell_size',	'unif_cell_shape',	'marg_adhesion',	'single_epith_cell_size',	'bare_nuclie',	'bland_chrom',	'norm_nuclie',	'mitosis',	'diagnosis']
#putting features on x axix
X = data[feature_coles]
#putting result on y axis
y = data['diagnosis']

#splitting data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=5)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_predlog = logreg.predict(X_test)
#calculating and printing the accuracy
print("Accuracy (%): ")
print(metrics.accuracy_score(y_predlog, y_test))

#to enter your own values modify the following lines of code
logreg.fit(X, y)
sample = [19.17,	24.8,	132.4,	1123,	0.0974,	0.2458,	0.2065,	0.1118,	0.2397,	0.078,	0.9555,	3.568,	11.07,	116.2,	0.003139,	0.08297, 0.0889,	0.0409,	0.04484,	0.01284,	20.96,	29.94,	151.7,	1332,	0.1037,	0.3903,	0.3639,	0.1767,	0.3176,	0.1023]
f_sample = np.reshape(sample, (1, -1))#pass an array of your readings in argument example predict(sample)
predlog = logreg.predict(f_sample)
print("\nPredicted result : ")
print(predlog) #printing the array with predicted result

if predlog[0] == 1:     # 1 donates Malignant type of breast cancer
    print("\nType of Breast cancer : Malignant")
elif predlog[0] == 0:      # 0 donates Benign type of breast cancer
    print("\nType of Breast cancer : Benign")