# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 18:22:42 2018

@author: Jiajie Hu
"""
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import f_classif, f_regression
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.cluster import DBSCAN
from pyproj import Proj, transform
from scipy.spatial import distance
from collections import Counter
from sklearn.model_selection import GridSearchCV
from DENCLUE import DENCLUE
from gridbscan import GRIDBSCAN
from sklearn.metrics import classification_report, confusion_matrix, roc_curve 
from sklearn import tree


def loadData():
    df = pd.read_csv('BSM.csv')
    df = df[['Latitude','Longitude','Speed','Ax','Ay','Az','JOIN_FID','PASER_12']]
    df = df[df.Ay != 2001]
    return df

def createRoadData(roadID,df):
    '''
    Description:
        create data point features for each road, each data point contain 1s received data (10). 
    Parameters:
        roadID-ID for each raod
        df-dataframe of raw data
    Returns:
        matrix-contain 8 features for each data point
    '''
    road = df[df.JOIN_FID == roadID]
    ID = road.iloc[0,6] 
    PASER = road.iloc[0,7] 
    road = road[['Latitude','Longitude','Speed','Ax','Ay','Az']]
    road = road.values 
  
    n = len(road) # number of whole lines
    m = int(n/10) #number of data points
    
    if m == 0:
        matrix = []
        return matrix
    
    matrix = np.zeros((m,12))
    mIndex = 0 # row index for matrix    
    count = 0 
    for i in range(n):
        count +=1
        
        if count == 1:
            temp = road[i,:]
            continue
        temp = np.vstack([temp, road[i,:]])
        
        if count ==10:
            index = list(temp[:,5]).index(max(temp[:,5]))
            Lat = temp[index,0]
            Lon = temp[index,1]
            Ms = np.mean(temp[:,2])
            Mx = np.mean(temp[:,3])
            My = abs(np.mean(temp[:,4]))
            Mz = np.mean(temp[:,5])
            SDs = np.std(temp[:,2])
            SDx = np.std(temp[:,3])
            SDy = np.std(temp[:,4])
            SDz = np.std(temp[:,5])
            matrix[mIndex,:] = [ID,PASER,Lat,Lon,Ms,SDs,Mx,SDx,My,SDy,Mz,SDz]
            mIndex += 1
            count = 0
    return matrix

def createRoadLabel(roadDataMat):
    nrow = len(roadDataMat)
    SDz = roadDataMat[:,-1]
    label = np.zeros((nrow,))
    
    if len(label[SDz>=45]) != 0:
        label[SDz>=45] = 1 
        pothole = roadDataMat[label == 1]
        clusterLabel = clusterPothole(pothole)
    
        if len(clusterLabel[clusterLabel!=-1]) != 0:
            clusterLabel[clusterLabel!=-1] = 1
    
        if len(clusterLabel[clusterLabel==-1]) != 0:
            clusterLabel[clusterLabel==-1] = 0    
    
        label[label==1] = clusterLabel
    
    roadData = np.column_stack((roadDataMat,label))
    return roadData


def createDataMat(df):
    roadID = set(df.JOIN_FID)
    count = 0

    for i in roadID:
        
        if i == 1599: #too big
            continue
        
        count += 1
        temp = createRoadData(i,df)

        if len(temp) == 0:
            continue
        
        temp = createRoadLabel(temp)
        if count == 1:
            returnMat = temp
            continue
        
        returnMat = np.vstack([returnMat, temp])
    
    return returnMat

def balanceDataMat(dataMat, dataLabel):
    potholeIndex = np.argwhere(dataLabel == 1) 
    numPothole = len(potholeIndex) 
    potholeIndex = [int(potholeIndex[i,:]) for i in range(numPothole)]       
    smoothIndex = np.argwhere(dataLabel == 0)
    numSmooth = len(smoothIndex)
    smoothIndex = [int(smoothIndex[i,:]) for i in range(numSmooth)] 
    smoothIndex = random.sample(smoothIndex, numPothole)
    dataMatNew = dataMat[potholeIndex+smoothIndex,:]
    dataLabelNew = dataLabel[potholeIndex+smoothIndex,]    
    return dataMatNew, dataLabelNew

    
def featureSelection(dataMat, dataLabel):   
#    model = SelectFpr(score_func=f_classif, alpha=0.05)
#    model.fit(dataMat,dataLabel)
#    dataNew = model.transform(dataMat)
#    return model.pvalues_, dataNew
    rf = RandomForestClassifier()
    rf.fit(dataMat, dataLabel)
    score = rf.feature_importances_
    return score


def svmClassifier(dataMat, dataLabel):
    param_grid = {'C':[1,10,100],'gamma':[0.001, 0.01, 0.1]}
    svr = SVC()
    clf = GridSearchCV(svr, param_grid)
    clf.fit(dataMat, dataLabel)        
    classifierResult = clf.predict(dataMat)
    confusion_matrix(trainingLabel,classifyResult)  
    classification_report(trainingLabel,classifyResult) 
#    roc = roc_curve(svr.predict_proba(X), y) prob should be true 
#    from sklearn.cross_validation import cross_val_score
#    cross_val_score(model, X, y, scoring = 'roc_auc')
    #joblib.dump(GDB, "GDB_model.m")
    #clf = joblib.load("SVM_model.m")
    #clf.best_score_,clf.best_estimator_,clf.best_params_, clf.cv_results_   
    return classifierResult


def kCluster(roadDataMat):
    VerticalAcc = roadDataMat[:,-2:]
    kmeans = KMeans(n_clusters=2)
    kmeans = kmeans.fit(VerticalAcc)
    labels = kmeans.predict(VerticalAcc)
    C = kmeans.cluster_centers_
    return labels,C

   
def linearRegression():
    data = {'Ms':trainingMat[:,0],'SDs':trainingMat[:,1],'Mx':trainingMat[:,2],'SDx':trainingMat[:,3],
            'My':trainingMat[:,4],'SDy':trainingMat[:,5],'Mz':trainingMat[:,6],'SDz':trainingMat[:,7],'Label': trainingLabel}
    trainingDF = pd.DataFrame(data)
    formula = 'Label~T+G+T:G'                           
    trainingMat_scale = preprocessing.scale(trainingMat)                                           
    model = sm.OLS(trainingLabel,trainingMat_scale).fit()                  
    results = anova_lm(model)
    model.summary()
    x1 = trainingMat[:,0]
    y = trainingLabel
    plt.scatter(x1, y, c='black', s=7)
    y_pred = model.predict(trainingMat[:,6:8])
    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))

   
def clusterPothole(dataMat, classifyResult):   
    pothole = dataMat[classifyResult == 1]
    Lon = pothole[:,3]
    Lat = pothole[:,2]
    outProj = Proj(init='epsg:3857')
    inProj = Proj(init='epsg:4326')
    x,y = transform(inProj,outProj,Lon,Lat)
    potholeCoord = np.column_stack((x,y))    
    GDB=GRIDBSCAN(gridsize=5,min_samples=5)
    GDB.fit(potholeCoord)
    GDB_labels = GDB.labels_
    
    filterLabel = np.asarray(GDB_labels)
    filterLabel[filterLabel != -1] = 1
    filterLabel[filterLabel == -1] = 0
    a=classifyResult
    a[a==1] = filterLabel
    
    clusterMat = np.column_stack((Lon,Lat,GDB_labels))
    clusterID = set(GDB_labels)
    clusterID.remove(-1)
    nrow = len(clusterID)
    clusterInfo = np.zeros((nrow,4))
    index = 0
    for i in clusterID:
        temp = clusterMat[clusterMat[:,2] == i,:]
        density = len(temp)
        Lon = np.mean(temp[:,0])
        Lat = np.mean(temp[:,1])
        clusterInfo[index,:] = [i,Lon,Lat,density]
        index += 1
    cluster_df = pd.DataFrame(clusterInfo,columns = ['ID','Lon','Lat','Density'])
    #cluster_df.to_csv('cluster_df.csv',index=False) 
    
    gridLabel = np.asarray(GDB.index)[:,1]
    PASER = pothole[:,0:2]
    gridMat = np.column_stack((Lon,Lat,PASER))
    hotGrid = np.asarray(GDB.hot_grid)[:,1] 
    nrow = len(hotGrid)
    gridInfo = np.zeros((nrow,5))
    index = 0
    for i in hotGrid:
        temp = gridMat[gridLabel == i,:]
        density = len(temp)
        Lon = np.mean(temp[:,0])
        Lat = np.mean(temp[:,1])
        roadList = list(temp[:,2])
        road = max(set(roadList), key=roadList.count)
        rate = (temp[temp[:,2] == road, 3])[0]
        gridInfo[index,:] = [Lon,Lat,density,road,rate]
        index += 1
    
    grid_df = pd.DataFrame(gridInfo,columns = ['Lon','Lat','Density','RoadID','PASER'])
    #grid_df.to_csv('grid_df.csv',index=False) 
          
#    cluster = GDB.cluster_set  
#    clusterLabel = DBSCAN(eps = 5, min_samples = 4).fit_predict(potholeCoord)    
#    DEN = DENCLUE(min_density = 5)
#    DEN.fit(potholeCoord)
#    cluster_info = den.cluster_info_
#    DEN_labels = den.labels_
#    a = potholeCoord[0,:]
#    b = potholeCoord[2,:]
#    dst = distance.euclidean(a, b)
    return classifyResult, clusterInfo, gridInfo

def calculateAccuracy(trueLabel, classifyLabel):
    cm = confusion_matrix(trueLabel, classifyLabel)
    cr = classification_report(trueLabel, classifyLabel)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    return cm, cr

def createRoadGridInfo(gridInfo):
    
    roadInfo = pd.read_excel('AA_roadrate.xlsx')
    potholeRoadList = set(gridInfo[:,3])
    allRoadList = set(trainingMat[:,0])
    Num_potholeRoad = len(potholeRoadList)
    Num_allRoad = len(allRoadList)
    roadInfo = roadInfo.loc[roadInfo['OBJECTID_1'].isin(allRoadList)]
       
    roadGridInfo = np.zeros((Num_allRoad,6))
    index = 0
    
    for i in allRoadList:
        length = int(roadInfo.loc[roadInfo['OBJECTID_1'] == i,'Shape_Leng'])
        PASER = int(roadInfo.loc[roadInfo['OBJECTID_1'] == i,'PASER_12'])
        
        if PASER >= 8:
            condition = 4
        elif PASER <= 7 & PASER >= 6:
            condition = 3
        elif PASER <= 5 & PASER >= 4:
            condition = 2
        else:
            condition = 1
                
        if i in potholeRoadList:  
            temp = gridInfo[gridInfo[:,3] == i,:]
            NumGrid = len(temp)
            Numpothole = sum(temp[:,2])
        else:
            NumGrid = 0
            Numpothole = 0
        
        roadGridInfo[index,:] =[i,NumGrid,Numpothole,length,condition,PASER]    
        index += 1
    
    gridbyLength = roadGridInfo[:,1]/roadGridInfo[:,3]*100
    potholebyLength = roadGridInfo[:,2]/roadGridInfo[:,3]*100
    roadGridInfo = np.column_stack((roadGridInfo,gridbyLength,potholebyLength))
    roadGridInfo_df = pd.DataFrame(roadGridInfo,columns = ['ID','NumGrid','Numpothole','Length','Condition'
                                                           ,'PASER','Grid/Length','Pothole/Length'])
    #roadGridInfo_df.to_csv('roadGridInfo.csv',index=False)
    return roadGridInfo_df

def createIRIProxyForRoad(df):
    roadInfo = pd.read_excel('AA_roadrate_raw.xlsx')
    roadID = set(df.JOIN_FID)
    Num_road = len(roadID)
    roadRMS = np.zeros((Num_road,9))
    index = 0
    
    for ID in roadID:
        road = df[df.JOIN_FID == ID]
        PASER_10 = int(roadInfo.loc[roadInfo['OBJECTID_1'] == ID,'PASER_10'])
        PASER_12 = int(roadInfo.loc[roadInfo['OBJECTID_1'] == ID,'PASER_12'])         
        PASER_14 = int(roadInfo.loc[roadInfo['OBJECTID_1'] == ID,'PASER_14'])
        PASER = PASER_12
        
        if PASER >= 8:
            condition = 4
        elif PASER <= 7 & PASER >= 6:
            condition = 3
        elif PASER <= 5 & PASER >= 4:
            condition = 2
        else:
            condition = 1
        
        length = int(roadInfo.loc[roadInfo['OBJECTID_1'] == ID,'Shape_Leng'])       
        n = len(road)
        Az = (road['Az'])*9.81  
        speed = road['Speed']
        RMS = np.sqrt(np.mean((Az/speed)**2))
        roadRMS[index] = [ID, n, condition, RMS, length, PASER_10, PASER_12, PASER_14, PASER]
        index += 1
    roadRMS_df = pd.DataFrame(roadRMS,columns = ['ID', 'Num_datapoint', 'RoadCondition', 'RMS', 'length', '10', '12', '14', 'PASER'])
    roadRMS_df.to_csv('roadRMS_df.csv',index=False)
    return roadRMS


def plotVerticalAcc(roadDataMat):
    VerticalAcc = roadDataMat[:,-2:]
    nrow = len(VerticalAcc)
    nsample = 30000
    index = random.sample(range(nrow), nsample)
    X = VerticalAcc[index,:]
    x1 = X[:,0]
    x2 = X[:,1]
    plt.scatter(x1, x2, c='black', s=7)
    
if __name__ == '__main__':
    
    df = loadData()    
    dataMat = createDataMat(df)       
    trainingMat, trainingLabel = balanceDataMat(dataMat[:,0:-1], dataMat[:,-1])    
    score = featureSelection(trainingMat[:,4:12], trainingLabel)   
    trainingMat_scale = preprocessing.scale(trainingMat[:,9:12])
    classifyResult = clf.predict(trainingMat_scale)    
    filterLabel, clusterInfo, gridInfo = clusterPothole(trainingMat, classifyResult)    
    roadGridInfo = createRoadGridInfo(gridInfo)
    roadRMS = createIRIProxyForRoad(df)
    
    x = np.asarray(roadRMS_dfcsv['RMS'])
    y = np.asarray(roadRMS_dfcsv['RoadCondition'])
    plt.scatter(x, y, c=y, s=7)    
    n =len(x)
    y_pred = np.zeros((n,))
    for i in range(n):
        if x[i] >= 2.671153:
            y_pred[i] = 1
        elif (x[i] < 2.671153) & (x[i] >= 1.84146):
            y_pred[i] = 2
        elif (x[i] < 1.84146) & (x[i] >= 1.165347):
            y_pred[i] = 3
        else:
            y_pred[i] = 4
    
    confusion_matrix(y,y_pred)
    classification_report(y,y_pred)
    roadRMS_dfcsv = roadRMS_dfcsv.assign(condition_pred=y_pred)
    roadRMS_dfcsv.to_csv('road_pred.csv',index=False)
    
    score1 = featureSelection(roadGridInfocsv[['Grid/Length','Pothole/Length']], roadGridInfocsv['Condition'])  
    
    
    
    
    
    
    
    

    
    svr = SVC()
    svr.fit(roadGridInfocsv['Grid/Length'], roadGridInfocsv['Condition'])        
    classifierResult = clf.predict(dataMat)
    svr.score(roadGridInfocsv[['Grid/Length','Pothole/Length']], roadGridInfocsv['Condition'])
    confusion_matrix(trainingLabel,classifyResult)  
    classification_report(trainingLabel,classifyResult) 
    

    