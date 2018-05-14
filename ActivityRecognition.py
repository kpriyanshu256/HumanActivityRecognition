import pandas as pd
import numpy as np
import pickle

def get_data():
    file_X_train='UCI HAR Dataset/train/X_train.txt'
    file_y_train='UCI HAR Dataset/train/y_train.txt'
    file_X_test='UCI HAR Dataset/test/X_test.txt'
    file_y_test='UCI HAR Dataset/test/y_test.txt'
    
    X_train=pd.read_csv(file_X_train,header=None,sep='\n')
    y_train=pd.read_csv(file_y_train,header=None,sep='\n')
    X_test=pd.read_csv(file_X_test,header=None,sep='\n')
    y_test=pd.read_csv(file_y_test,header=None,sep='\n')
    
    X_train=process_data(X_train)
    X_test=process_data(X_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)

    return X_train,y_train,X_test,y_test
    

def process_data(df):
    data=[]
    for i in range(df.shape[0]):
        t=df.iloc[i]
        t=t[0].split()
        t=[ float(j) for j in t]
        data.append(t)  
    data=np.array(data)
    return data
        

if __name__=='__main__':
    X_train,y_train,X_test,y_test=get_data()
    data={}
    data['X_train']=X_train
    data['y_train']=y_train
    data['X_test']=X_test
    data['y_test']=y_test
    
    save=open('har_data.pkl','wb')
    pickle.dump(data,save)
    save.close()