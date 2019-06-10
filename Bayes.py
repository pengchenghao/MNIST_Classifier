from numpy import *
from MNISTData import *
from PCA import *
from numpy.linalg import inv,det

import collections



def test(Num):
    
    #降到15维
    D=15
    trainingData,trainingLabel,testData,testLabel=getData()
    trainingData=trainingData[:Num,:]
    trainingLabel=trainingLabel[:Num]
    trainingData,DimReduVct,PCAmean=PCA(trainingData,D)

    #下标分类
    indexOf=[None for i in range(10)]
    for i in range(10):
        indexOf[i]=argwhere(trainingLabel==i)
    #均值
    meanOf=[None for i in range(10)]
    for i in range(10):
        meanOf[i]=mean(trainingData[indexOf[i],:],axis=0)
    #协方差矩阵
    covarianceOf=[None for i in range(10)]
    for i in range(10):
        temp=indexOf[i]
        temp.shape=-1
        covarianceOf[i]=cov(trainingData[temp].T)

    testData=(testData-PCAmean)
    testData=dot(testData,DimReduVct)
    #识别（10000个)
    hit=0
    for sample in range(10000):
        testPoint=testData[sample,:]
        possibilityOf=[0 for i in range(10)]
        for i in range(10):
            possibilityOf[i]=exp(-0.5*dot(dot((testPoint-meanOf[i]),\
        inv(covarianceOf[i])),(testPoint-meanOf[i]).T))/sqrt(det(covarianceOf[i]))
                                
        guest=argmax(possibilityOf)
        testlabel.append(testLabel[sample])
        if guest==testLabel[sample]:
            hit+=1
        else:
           test_error.append(testLabel[sample]) #
   # print("\n test complete!")
   # print(hit,"hit")
    print("\n trainingData:",Num)
    print("correct rate:{:.2f}%".format((hit/10000.0)*100))


##------统计每个数字分类正确的概率------- 

def cal_correct(test_error,test_data):
    b=collections.Counter(test_error)
    for m in b:
        print(m,b[m])
    q=collections.Counter(test_data)

    for n in q:
        print(n,'\t',1-b[n]/q[n])
        
if __name__=="__main__":
    #train_num=[500,1000,1500,5000,10000,20000,30000,40000,50000,60000]
    test_error=[]
    testlabel=[]
    print("Bayes")
    #for i in train_num:
     #   test(i)
    test(500)
    cal_correct(test_error,testlabel)

   
