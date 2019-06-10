from struct import*
from numpy import *

def getData():
    #读入训练集
    trainingImageFile=open(r'F:\Pattern_Recognition\MNIST\train-images.idx3-ubyte','rb')
    trainingImageFile.read(16)
    trainingData=fromfile(trainingImageFile,dtype=uint8)
    trainingData.shape= -1,784
    #训练集数据标签切片
    trainingData=trainingData[0:60000,:]
    trainingImageFile.close()
    #读入训练标签集
    trainingLabelFile=open(r'F:\Pattern_Recognition\MNIST\train-labels.idx1-ubyte','rb')
    trainingLabelFile.read(8)
    trainingLabel=fromfile(trainingLabelFile,dtype=uint8)
    #训练集标签切片
    trainingLabel=trainingLabel[0:60000]
    trainingLabelFile.close()
    #读入测试数据
    testImageFile=open(r'F:\Pattern_Recognition\MNIST\t10k-images.idx3-ubyte','rb')
    testImageFile.read(16)
    testData=fromfile(testImageFile,dtype=uint8)
    testData.shape=-1,784
    testImageFile.close()
    #读入训练集标签
    testLabelFile=open(r'F:\Pattern_Recognition\MNIST\t10k-labels.idx1-ubyte','rb')
    testLabelFile.read(8)
    testLabel=fromfile(testLabelFile,dtype=uint8)
    testLabelFile.close()

    return trainingData,trainingLabel,testData,testLabel
