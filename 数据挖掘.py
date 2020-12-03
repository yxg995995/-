import pandas as pd
import numpy as np
# K-Means聚类算法
data=pd.read_csv(r'C:\Users\Windows 10\.spyder-py3\信用卡消费.csv')
data=np.array(data)
data1=data[:,2:6]
data2=data[:,7:]
data3=np.hstack((data1,data2))
data=pd.DataFrame(data3)
outputfile=r'D:\data_type.xls'
k=3
iteration=500
zscoredfile=r'D:\zscoreddata.xls'
data_zs=1.0*(data-data.mean())/data.std()
data_zs.to_excel(zscoredfile,index=False)
from sklearn.cluster import KMeans
model=KMeans(n_clusters=k,n_jobs=4,max_iter=iteration)
model.fit(data_zs)
r1=pd.Series(model.labels_).value_counts()
r2=pd.DataFrame(model.cluster_centers_)
r=pd.concat([r2,r1],axis=1)
r.columns=list(data.columns)+[u'类别数目']
print(r)
r=pd.concat([data,pd.Series(model.labels_,index=data.index)],axis=1)
r.columns=list(data.columns)+[u'聚类类别']
r.to_excel(outputfile)
def density_plot(data):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    p=data.plot(kind='kde',linewidth=2,subplots=True,sharex=False)
    [p[i].set_ylabel(u'密度') for i in range(k)]
    plt.xlabel('分解%s'%(i+1))
    plt.legend()
    return plt
pic_output=r'D:\pd_'
for i in range(k):
    density_plot(data[r[u'聚类类别']==i]).savefig(u'%s%s.png'%(pic_output,i))



