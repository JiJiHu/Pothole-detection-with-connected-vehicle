# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import Counter
## Read raw data
#df = pd.read_csv('RSE BSM.csv', usecols=[0,6,7,11,12,14,15,16],header=None)
#columns = ("RxDevice",  "Latitude", "Longitude", "Speed", "Heading", "Ax",	"Ay",	"Az")
#df.columns = columns

## Data filter based on coordinate and speed
#df = df[df.Speed >= 100]
#df = df[(-838030000 <= df.Longitude) & (df.Longitude <= -836760000)]
#df = df[(422280000 <= df.Latitude) & (df.Latitude <= 423240000)]
#df = df.drop(columns = ['Elevation','Heading'])
#df.describe
#df.to_csv('BSM.csv',index=False)

## Seperately Save to CSV
#reader = pd.read_csv('BSM.csv', chunksize=7510000)
#fileName = ["RSE BSM1.csv","RSE BSM2.csv","RSE BSM3.csv","RSE BSM4.csv","RSE BSM5.csv",
#            "RSE BSM6.csv","RSE BSM7.csv","RSE BSM8.csv","RSE BSM9.csv","RSE BSM10.csv"]
#n = 0
#for chunk in reader:
#       chunk.to_csv(fileName[n],index=False)      
#       n += 1
#reader = pd.read_csv("RSE BSM1.csv",iterator=True)
#a = reader.get_chunk(10)

## Trim spatial join data
#bsm1 = pd.read_csv('RSE_BSM1.csv')
#bsm2 = pd.read_csv('RSE_BSM2.csv')
#bsm3 = pd.read_csv('RSE_BSM3.csv')
#bsm4 = pd.read_csv('RSE_BSM4.csv')
#
#bsm1 = bsm1[['OBJECTID_1','RxDevice','Latitude','Longitude','Speed','Heading','Ax','Ay','Az','JOIN_FID','PASER_12']]
#bsm1.drop_duplicates(keep='first', inplace=True)
#bsm2 = bsm2[['OBJECTID_1','RxDevice','Latitude','Longitude','Speed','Heading','Ax','Ay','Az','JOIN_FID','PASER_12']]
#bsm2.drop_duplicates(keep='first', inplace=True)
#bsm3 = bsm3[['OBJECTID_1','RxDevice','Latitude','Longitude','Speed','Heading','Ax','Ay','Az','JOIN_FID','PASER_12']]
#bsm3.drop_duplicates(keep='first', inplace=True)
#bsm4 = bsm4[['OBJECTID_1','RxDevice','Latitude','Longitude','Speed','Heading','Ax','Ay','Az','JOIN_FID','PASER_12']]
#bsm4.drop_duplicates(keep='first', inplace=True)
#bsm = pd.concat([bsm1,bsm2,bsm3,bsm4])
#
#bsm1.to_csv('RSE_BSM1.csv',index=False)
#bsm2.to_csv('RSE_BSM2.csv',index=False)
#bsm3.to_csv('RSE_BSM3.csv',index=False)
#bsm4.to_csv('RSE_BSM4.csv',index=False)
#bsm.to_csv('BSM.csv',index=False)

#u = np.array([[1,2],[3,4]])
#m = u.tolist()   #转换为list
#m.remove(m[0])    #移除m[0]
#m = np.array(m)    #转换为array

## Road rate analysis
df = pd.read_csv('BSM.csv')
df.describe
df1 = df[0:100000]



df1 = df[(df.Az >= 45) & (df.Az <= 55)]
df1 = df1.drop(columns = ['Elevation','Heading'])
df1670 = df[(df.JOIN_FID == 1670) & (df.Az >= 75)] #poor
df5 = df[df.JOIN_FID == 5]
df1641 =df[df.JOIN_FID == 1641]
df522 = df[(df.JOIN_FID == 522)]

bsm1 = pd.read_csv('BSM_10_1_s_join.csv')
bsm1.describe
a = bsm1[bsm1.Az <=-0.5]
a = a[['Az','PASER_12','JOIN_FID']]
highZ = sorted(list(Counter(a.PASER_12).items()))
wholeZ = sorted(list(Counter(bsm1.PASER_12).items()))
np.divide(highZ,wholeZ)

Counter(set(bsm1[['PASER_12','JOIN_FID']]))



