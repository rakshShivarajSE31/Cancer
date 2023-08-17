#this shows the classification of cancer and it's %value


import numpy as np            # Import necessery modules
import pandas as pd
from sklearn import neighbors, model_selection

df = pd.read_excel('bcancer.xlsx')
df.replace('?', -99999, inplace=True)  # create an outlier for missing values such that many algorithms detects that and penalises it
df.drop(['id'], 1, inplace=True)# Remove Unnecessery data from the input xcel file before modelling it.

X= np.array(df.drop(['diagnosis'], 1))
y= np.array(df['diagnosis'])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.25)


from sklearn.neighbors import KNeighborsClassifier
             # Use Knn classification model
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)



             #logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
logistic_regression = LogisticRegression().fit(X_train, y_train)
print("training set score is:", logistic_regression.score(X_train, y_train)*100)
print("test score is ", logistic_regression.score(X_test, y_test)*100)



from sklearn.metrics import accuracy_score
from sklearn import tree
clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(X, y)
pred = clf1.predict(X_test)
#clf1.score(iris.data, iris.target)
print("descion tree", clf1)
print("pred", pred)
#accuracy
print("decsion tree accuracy is :", accuracy_score(y_test, clf1.predict(X_test)))






                    #random forest
from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(n_estimators=100)
clf2.fit(X_train,y_train)
y_pred = clf2.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Random forest Accuracy is:",metrics.accuracy_score(y_test, y_pred))







from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
# making predictions on the testing set
y_pred = gnb.predict(X_test)
# comparing actual response values (y_test) with predicted response values (y_pred)
#accuracy
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)





from sklearn.metrics import confusion_matrix
con = confusion_matrix(y_test, y_pred)
print(con)




import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(con, annot = True)
plt.xlabel('x_axis')
plt.ylabel('y_axis')
plt.title('seaborn')
plt.show()

