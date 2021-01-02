import numpy as np
import pandas as pd
import operator
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
""" ScikitLearn is Only Used for Creation of the Dataset and the Splitting of the dataset"""


""" KNN from Basics """
class KNN:
    def squared_error_distance(self,testing_data,training_data,test_features):
        sq_d=0
        for i in range(0,test_features):
            ##Squared_Error_Distances
            ##Of Each Features
            sq_d+=((testing_data[i]-training_data[i])**2)
        sq_d=sq_d**0.5
        return sq_d
    def model(self,training_set,testing_set,K):
        #taking the dimensions into account
        test_size=testing_set.shape[0]
        test_features=testing_set.shape[1]
        test_work=[]
        train_size=training_set.shape[0]
        """ Each Testing Set """
        ###Storing Distances of Each Testing Sets from the Training Sets
        for j in range(0,test_size):
            ##Dictionary of Distance of Each testing set
            ##From all the Training Sets
            distance_dict=dict()
            for i in range(0,train_size):
                #dd=squared_error_distance(testing_set.iloc[j],training_set.iloc[i],test_features)
                dd=0
                for k in range(0,test_features):
                    dd+=((testing_set.iloc[j][k]-training_set.iloc[i][k])**2)
                dd=dd**0.5
                distance_dict[i]=dd
                ###Sorting the Items
            sorted_distance_list=sorted(distance_dict.items(),key=operator.itemgetter(1))
            neighbours=[]
            for i in range(0,K):
                ##Appending the Nearest K Points from the Particular Test Set
                ##i.e. from the jth Testing Set
                neighbours.append(sorted_distance_list[i][0])
            votes=dict()
            for i in range(0,len(neighbours)):
                ## Response is the Label
                ##That the Particular Point is classified
                ##in the Training Set
                response=training_set.iloc[neighbours[i]][-1]
                if response not in votes.keys():
                    votes[response]=1
                else:
                    votes[response]+=1
            ##Sort in Descending Order
            sorted_votes_list=sorted(votes.items(),key=operator.itemgetter(1),reverse=True)
            ##Store the Particular Label/Class
            ##To which our jTH test set is nearer to
            test_work.append([int(sorted_votes_list[0][0]),neighbours])
            ##Return the Predicted Labels and the Neighbours
            ##For all the Testsets
        self.test_result=test_work
        
    def accuracy_model(self,y_true,y_pred):
        self.accuracy=accuracy_score(y_true,y_pred)*100
        
    def f1_model(self,y_true,y_pred):
        self.f1=f1_score(y_true,y_pred,average=None)

""" Testing Phase """
""" Let's try on spliiting the main data set to 140-10 """

""" Loading of Datasets """
#iris=datasets.load_iris()
#df_iris=pd.DataFrame(data=iris.data,columns=iris.feature_names)
#df_iris['Class']=iris.target
#print(df_iris.head())
training_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_y = pd.read_csv('gender_submission.csv')
#Randomly Shuffling the Datasets
#df_iris=df_iris.sample(frac=1).reset_index(drop=True)

#Spliting of the Datasets into Training Sets and Testing Sets
training_set=training_df.iloc[:,[2,4,6,7,1]]
training_set = pd.get_dummies(training_set)
testing_set=test_df.iloc[:,[1,3,5,6]]
testing_set = pd.get_dummies(testing_set)

#print(testing_set.Class)

Kmax=7
basic_accuracys=dict()
basic_f1_scores=dict()
#y_true=testing_set.Survived.values
y_true=test_y.iloc[:,1].values
knn=KNN()
for i in range(1,Kmax+1,2):
    knn.model(training_set,testing_set,i)
    test_work=knn.test_result
    y_pred=[test_work[i][0] for i in range(0,len(test_work))]
    knn.accuracy_model(y_true,y_pred)
    basic_accuracys[i]=knn.accuracy
    knn.f1_model(y_true,y_pred)
    basic_f1_scores[i]=knn.f1

plt.plot(basic_accuracys.keys(),basic_accuracys.values(),color='green')
plt.xlabel('No of Centroids')
plt.ylabel('Accuracy Score(without using Libraries)')
plt.show()

plt.plot(basic_f1_scores.keys(),basic_f1_scores.values(),color='pink')
plt.xlabel('No of Centroids')
plt.ylabel('F1 Score(Without Using Libraries)')
plt.show()
