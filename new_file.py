import pandas as pd
t_shirt = pd.read_excel('price.xlsx')
print(t_shirt.head())









import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Importing the dataset

df = pd.read_excel('bcancer.xlsx')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

x = np.array(df.drop(['classes'], 1))
y = np.array(df['classes'])

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

knn = []
#for i in range(1, 21):
classifier = KNeighborsClassifier(n_neighbors=3)
trained_model = classifier.fit(x_train, y_train)
trained_model.fit(x_train, y_train)


# Predicting the Test set results

y_pred = classifier.predict(x_test)
# Making the Confusion Matrix
cm_KNN = confusion_matrix(y_test, y_pred)
print(cm_KNN)
print("Accuracy score of train KNN")
print(accuracy_score(y_train, trained_model.predict(x_train)) * 100)

print("Accuracy score of test KNN")
print(accuracy_score(y_test, y_pred) * 100)

knn.append(accuracy_score(y_test, y_pred) * 100)

plt.figure(figsize=(12, 6))
plt.plot(range(x, y), knn, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
#return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]
plt.title('Accuracy for different  K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.show()
return(np.frombuffer(bytestream.read(4), dtype=dt)[0])