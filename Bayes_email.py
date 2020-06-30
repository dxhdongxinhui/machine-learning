# -*- coding: utf-8 -*-

import numpy as np
import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random

#函数功能：将字符串转化为字符列表
def textParse(bigString):
    listOfTokens = re.split(r'\W+', bigString)  #用非字符，非数字作为切分标志
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  #除了单个字母，其它单词变成小写

#函数功能：文件加载
#返回值：
#docList：每一封邮件转化成字符串列表之后组成的列表
#classList：对应的邮件标记，1表示垃圾邮件，0表示非垃圾邮件
def dataload():
    docList = []
    classList = []
    for i in range(1, 26):
        wordList = textParse(open('data/spam/%d.txt' % i, 'r').read())  #读取每个垃圾邮件，将字符串转换成字符串列表
        docList.append(wordList)
        classList.append(1)  #1表示垃圾邮件
        wordList = textParse(open('data/ham/%d.txt' % i, 'r').read())  #读取每个非垃圾邮件，将字符串转换成字符串列表
        docList.append(wordList)
        classList.append(0)  #0表示非垃圾邮件
    return docList, classList


#函数功能：文件随机拆分
#返回值：
#trainingSet：作为训练集的邮件的索引值表
#testSet：作为测试集的邮件的索引值表
def random_split():
    trainingSet = list(range(50)) #训练集的索引值的列表
    testSet = []  #测试集的索引值的列表
    for i in range(20):  #随机选取30封邮件作为训练集，20封邮件作为测试集
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    return trainingSet, testSet

#函数功能：生成不重复的词汇表
#参数：
#docList：每一封邮件转化成字符串列表之后组成的列表
#返回值：
#vocabSet：不重复的词汇表
def createVocabList(docList):
    vocabSet = set([])
    for document in docList:
        vocabSet = vocabSet | set(document)  #取并集来保证不重复
    return list(vocabSet)

#函数功能：将字符串列表转化为0，1组成的列表（根据词汇表）
#参数：
#vocabList：词汇表
#inputSet：字符串列表
#返回值:
#returnVec：0，1组成的列表
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)               #创建一个其中所含元素都为0的向量
    for word in inputSet:                          #遍历每个词条
        if word in vocabList:                      #如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

#函数功能：训练函数，计算垃圾邮件以及非垃圾邮件下的条件概率以及先验概率
#参数：
#trainMat：训练文档矩阵
#trainClasses：训练类别标签向量
#返回值：
#p0Vect：非垃圾邮件下的条件概率列表
#p1Vect：垃圾邮件下的条件概率列表
#pAbusive：先验概率（垃圾邮件的出现频率）
def trainNB0(trainMat, trainClasses):
    numTrainDocs = len(trainMat)
    numWords = len(trainMat[0])
    pAbusive = sum(trainClasses) / float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)  #创建numpy.ones数组,词条出现数初始化为1,拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0  #分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainClasses[i] == 1:  #统计垃圾邮件下的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:                     #统计非垃圾邮件下的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)   #取对数，防止下溢出
    return p0Vect, p1Vect, pAbusive

#函数功能：分类器
#参数：
#vec2Classify：待分类的词条列表
#p0Vect：非垃圾邮件下的条件概率列表
#p1Vect：垃圾邮件下的条件概率列表
#pAbusive：先验概率（垃圾邮件）
#返回值：
#0：属于非垃圾邮件类
#1：属于垃圾邮件类
def classifyNB(vec2Classify, p0Vect, p1Vect, pAbusive):
    p1 = sum(vec2Classify*p1Vect)+np.log(pAbusive)
    p0 = sum(vec2Classify*p0Vect)+np.log(1.0-pAbusive)
    if p1 > p0:
        return 1
    else:
        return 0

#函数功能：绘制混淆矩阵
#参数：
#confusionMatrix：利用原标签和预测标签列表生成的混淆矩阵
def cm_plot(confusionMatrix):
    cm = confusionMatrix
    plt.figure()
    plt.matshow(cm, cmap = plt.cm.Reds)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix')
    plt.show()

#主函数
def main_():
    docList, classList = dataload()
    vocabList = createVocabList(docList)  # 创建词汇表，不重复
    trainingSet, testSet = random_split()
    trainMat = []
    trainClasses = []  # 创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet:  # 遍历训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))  # 将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])  # 将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))  # 训练朴素贝叶斯模型
    Pre_Result = []
    test_true = []
    for docIndex in testSet:  # 遍历测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])  # 测试集的词集模型
        Pre_Result.append(classifyNB(np.array(wordVector), p0V, p1V, pSpam))
        test_true.append(classList[docIndex])
    confusionMatrix = confusion_matrix(test_true, Pre_Result) #得出混淆矩阵
    cm_plot(confusionMatrix)
    accuracy_rate = (confusionMatrix[0][0] + confusionMatrix[1][1]) / len(testSet)
    return accuracy_rate

#交互过程
if __name__ == '__main__':
    accuracy_rate = []
    times = int(input("输入训练次数:\n"))
    for i in range(times):
        accuracy_rate.append(main_())
    accuracy_rate = np.array(accuracy_rate)
    print('accuracy_rate: %f' %(accuracy_rate.mean()))