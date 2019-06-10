#PCA(主成分分析）降维算法
from numpy import *
from MNISTData import *
def PCA(trainingData,k=2):
    means=mean(trainingData)
    covariance=cov(trainingData.T)
    scatterMatrix=(covariance.shape[0]-1)*covariance
    eigVal,eigVct=linalg.eig(scatterMatrix)
    topK=argsort(eigVal)[-k:]
    DimReduVct=eigVct[:,topK]
    principal=(trainingData-means)
    principal=dot(principal,DimReduVct)
    return principal,DimReduVct,means


if __name__=="__main__":
    m,n,x,y=getData();
    print(PCA(m,20))
