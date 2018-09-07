#IMPORTING MODULES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Reading Training & Testing Data
train_df= pd.read_csv("D://ML bvp//Data//Next Number//train.csv",nrows=74)
test_df = pd.read_csv("D://ML bvp//Data//Next Number//test.csv",nrows=74)
ANS = []
X = []
Y = []


#Approach to Problem
for seq in train_df["Sequence"]:
    
    SEQ=seq.split(',')
    for i in range(0,len(SEQ) -1 ):
        x = float(SEQ[i])
        y = float(SEQ[i+1])
        X.append(x)
        Y.append(y)
​
X=np.array(X).astype(np.float)
X=X.reshape(-1,1)
Y=np.array(Y).astype(np.float)


#Splitting of Data into Training & Testing Data
import sklearn.model_selection
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,Y,test_size=0.3)


#Support Vector Regressor (SVR)
from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf',gamma=0.1)
svr_rbf.fit(x_train,y_train)
print("SVR Accuracy-->",svr_rbf.score(x_test,y_test))


#Random Forest Regressor
import sklearn.ensemble
rf = sklearn.ensemble.RandomForestRegressor()
rf=rf.fit(x_train,y_train)
print("Random Forest Accuracy-->",rf.score(x_test,y_test))


#Prediction of next number in test sequence
for seq1 in test_df["Sequence"]:
    SEQ1=seq1.split(',')
    ans=svr_rbf.fit(X,Y).predict(float(SEQ1[len(SEQ1) - 1]))
    if ans != np.inf and ans != -np.inf and np.isnan(ans) == False:
        ANS.append(int(ans))
    else:
        ANS.append(int(0))


#Writing the predicted data to CSV file
submission =pd.DataFrame({"Id":test_df["Id"],"Last Number":np.array(ANS)})
submission.to_csv("D://ML bvp//Project Integer Sequence//final_submission.csv", index=False)