from sklearn.svm import SVC
from MNISTData import *
import random
import warnings

import collections
def test(Num):
#if __name__=="__main__":
    warnings.filterwarnings("ignore",category=FutureWarning,module="sklearn",lineno=193)
    trainingData,trainingLabel,testData,testLabel=getData()
    hit=0
    im=trainingData[0:Num,:]
    label=trainingLabel[0:Num]
    
    test=testData
    test_label=testLabel

    train_idx=list(range(Num))
    random.shuffle(train_idx)
    im_sample=im[train_idx]
    label_sample=label[train_idx]

    test_idx=list(range(10000))
    random.shuffle(test_idx)
    test_sample=test[test_idx]
    test_label_sample=test_label[test_idx]

    clf=SVC(kernel='poly')
    clf.fit(im_sample,label_sample)
    
    score=clf.score(test_sample,test_label_sample)
    x=clf.predict(test_sample)
    for i in range(10000):
        test_data.append(test_label_sample[i])
        if x[i]!=test_label_sample[i]:
            test_error.append(test_label_sample[i])
   
    
    
    #if test_label_sample!=clf.predict(test_sample):
     #   test_error.append(test_sample)
    #print("test complete!")
    #print("score:{:.6f}".format(score))
    #print("SVM:")
    
    print("\nTrainingData:",Num)
    print("Correct Rate:{:.2f}%".format(score*100))


 ##------统计每个数字分类正确的概率------- 

def cal_correct(test_error,test_data):
    b=collections.Counter(test_error)
    for m in b:
        print(m,b[m])
    q=collections.Counter(test_data)

    for n in q:
        print(n,'\t',1-b[n]/q[n])
        
           
if __name__=="__main__":
    #train_num=[1000,5000,10000,20000,30000,40000,50000,60000]
    print("VCM")
    test_error=[]        #分类错误的数字标签
    test_data=[]         #所有的数字标签，用以统计测试集中每种数字的个数
    #for i in train_num:
    #	    test(i)
    test(500)
    cal_correct(test_error,test_data)
    
 

    
