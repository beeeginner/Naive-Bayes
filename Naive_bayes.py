import numpy as np
import scipy.sparse as sp
import scipy

class GuassainNaiveBayes:
    '''
    处理连续值的高斯分布朴素贝叶斯
    '''
    def __init__(self,epsilon = 1e-6 , threshold = 1e-2):

        self.class_num = None #类别数
        self.prior = None #先验概率P(Y)
        self.means = None #高斯分布均值
        self.variance = None #高斯分布标准差
        self.epsilon = epsilon #防止标准差为0加入正则化项
        self.threshold = threshold

    def fit(self,X:np.ndarray,y:np.ndarray)->None:
        '''
        训练器
        :param X:特征矩阵
        :param y:类别向量
        '''

        assert isinstance(X,np.ndarray),r'请输入ndarray对象'
        n_samples,n_features = X.shape
        self.class_num = len(np.unique(y))
        self.prior = np.zeros(self.class_num)
        self.means = np.zeros((self.class_num,n_features))
        self.variance = np.zeros((self.class_num,n_features))

        #计算先验概率
        for i in range(self.class_num):
            #y == 1 得到一个逻辑数组，sum可以求出所有true的数目
            self.prior[i] = np.sum(y == i) / n_samples

        #计算高斯分布
        for j in range(self.class_num):
            #直接逐行更新
            self.means[j] = np.mean(X[y == j],axis = 0) + self.epsilon
            #普通稠密矩阵
            self.variance[j] = np.var(X[y == j], axis=0) + self.epsilon

    def _pdf(self,x,u = 0, sigma2 = 0.01):
        return np.exp(-(x-u) ** 2/(2 * sigma2)) / (np.sqrt(2*np.pi) * sigma2 ** 0.5)

    def predict(self,X):
        """
            预测
            :param X: 特征矩阵，shape为(n_samples, n_features)
            :return: 预测结果，shape为(n_samples,)
        """

        assert isinstance(X,np.ndarray),r'请输入ndarray对象'
        n_samples,n_features = X.shape
        if isinstance(X,scipy.sparse.csr_matrix):
            X = X.toarray(X)

        probs = np.zeros((n_samples,self.class_num))
        for i in range(n_samples):
            for j in range(self.class_num):
                #norm.pdf计算了当前样本所有特征对应的概率密度值
                #np.prod基于条件独立假设连乘所有概率
                p = self._pdf(X[i],self.means[j],self.variance[j])
                probs[i][j] = self.prior[j] * np.prod(p)
                probs[i][j] = np.nan_to_num(probs[i][j])

        return np.argmax(probs, axis=1)
