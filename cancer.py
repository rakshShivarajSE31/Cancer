from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
cancer = load_breast_cancer()
print("data", cancer.data)
#print("target", cancer.target)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.33)
logistic_regression = LogisticRegression().fit(x_train, y_train)
print("training set score is:", logistic_regression.score(x_train, y_train)*100)
print("test score is ", logistic_regression.score(x_test, y_test)*100)








