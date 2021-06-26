import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

Train_dataset = pd.read_csv('train.csv')
Test_dataset = pd.read_csv('test.csv')
X_train = Train_dataset.iloc[:, 53:-1]
X_train = pd.get_dummies(X_train)
y_train = Train_dataset.iloc[:, -1]
X_test = Test_dataset.iloc[:, 53:]
X_test = pd.get_dummies(X_test)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

cv = KFold(n_splits=7, random_state=1, shuffle=True)
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
scores = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=1)
print('Accuracy: %.5f (%.5f)'% (np.mean(scores), np.std(scores)))
y_pred = classifier.predict(X_test)
print(y_pred)
#ans_df = pd.DataFrame()
#ans_df['appno']=Test_dataset.iloc[:, 0]
#ans_df['importance']=y_pred
#ans_df.to_csv('DecisionTree_Answer.csv',index=False)