import pandas as pd
import numpy as np
data = pd.read_csv('model.csv')
from sklearn.feature_extraction import DictVectorizer


def getcategory():
    with open('category.csv') as f:
        data =  f.readlines()
        data = data[1:]
    
    def getDict(line):
        temp = line.strip().replace('"','').split(',')
        key = temp[0]
        value = temp[1:]
        lst = {}
        for s in value:
            s_lower = s.lower()
            if lst.get(s) is None:
                lst[s] = 1
            else:
                lst[s] = lst.get(s) + 1

        return  key, lst
    
    Dict = map(lambda x: getDict(x),data)

    vec = DictVectorizer()

    Features = vec.fit_transform(map(lambda x:x[1],Dict)).toarray()
    return map(lambda x,y:[x[0]] + y.tolist(),Dict,Features)

category = getcategory()
colname = ['x_%d'%i for i in range(len(category[0][1:]))]
category = pd.DataFrame(category,columns=['device_id']+colname)
modeldata = pd.merge(data,category,on=['device_id'],how='left')
modeldata = modeldata.fillna(dict(map(lambda x:(x,0),colname)))
del data,category

traindata = modeldata.copy()

traindata['gender_label'] = traindata['gender'].map(lambda x: 1 if x=='M' else 0)

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()



from sklearn.cross_validation   import train_test_split
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()
featureLables=['phone_brand','device_model','phone_brand_rate','device_model_rate',
               'events_rate','t_7-12','t_18-00','t_00-7','t_12-18',
               'latitude_max','longitude_max','latitude_min','longitude_min','latitude_mean','longitude_mean','squre']
#+colname
target = 'gender_label'
FeaturesDict = traindata[featureLables].to_dict('record')
Features = vec.fit_transform(FeaturesDict).toarray()


LabelsDict = traindata[target].to_frame().to_dict('record')

Labels = vec.fit_transform(LabelsDict).toarray().ravel()
Features_x = traindata[colname].values

Features_X =np.array(map(lambda x,y:np.array(x.tolist()+y.tolist()),Features,Features_x))
train_index = traindata[traindata['flag']=='train'].index
                        
test_index = traindata[traindata['flag']=='test'].index
del traindata
trainX,trainY = Features_X[train_index],Labels[train_index]
import xgboost
fit = xgboost.sklearn.XGBClassifier(learning_rate =0.3,
 n_estimators=500,
 max_depth=10,
 min_child_weight=4,
 gamma=1,
 subsample=0.7,
 colsample_bytree=0.8,
 reg_alpha =1,
 reg_lambda=0,
 objective= 'binary:logistic',nthread=-1, scale_pos_weight=1)
model = fit.fit(trainX,trainY,eval_set=[(trainX,trainY)],eval_metric='auc')

gender_label = model.predict(Features)

modeldata['gender'] = modeldata['gender'].fillna('N')
modeldata['gender_label'] = gender_label
modeldata['gender_label']  = modeldata['gender_label'].map(lambda x: 'M' if x==1 else 'F')
modeldata['gender'] = map(lambda x: x['gender_label'] if x['gender'] =='N' else x['gender'],modeldata[['gender','gender_label']].to_dict('records'))
modeldata.to_csv('modeldata.csv',index=False)
