"""
Restricted Boltzmann Machine
包含RBM的类  以及训练RBM+model/仅训练model的函数 训练RBM+model会保存训练过程中的似然值和预测准确率
"""
import time
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import scipy.sparse as sp
from scipy.special import expit  # logistic function
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import log_logistic
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets, metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import load_digits
from scipy.ndimage import shift
from sklearn.model_selection import train_test_split
class BernoulliRBM_all(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=256, *, learning_rate=0.1, batch_size=100,
                 n_iter=30, verbose=0, random_state=None, model_name):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state
        self.model_name = model_name
    def map_clip(self, old, coup, bias, a, b, noi):  # linear map with clipping
        "剪切函数"
        return np.clip(a * old + b * np.dot(coup, old) + .40 * b * bias + np.random.normal(0, noi, len(old)), -.4,
                       .4)
    def ising_energy(self, stuff, ad, bias):
        "ising energy求解"
        return -0.5 * 1 * np.dot(stuff, np.dot(ad, stuff)) - np.sum(np.dot(bias, stuff))

    def ising_sample(self, NN, n_hid):
        "超快速采样"
        global result
        global ens2
        global ad
        global chain
        global bias
        test = np.loadtxt(r'bla.txt')  # edge list
        N = int(np.max(test))  # number of nodes
        # negat=np.abs(np.identity(N)-1)
        ad = np.zeros((N, N))  # adjacency matrix
        for i in range(0, len(test)):  # initialize
            ad[int(test[i, 0]) - 1, int(test[i, 1]) - 1] += (test[i, 2])
        bias = np.zeros(N)
        for i in range(0, N):
            bias[i] += ad[i, i]
            ad[i, i] = 0
        # parameters for alpha and beta
        a = 0.9
        b = 0.1
        noise_strength = 0.12
        chain = np.zeros(N)
        ens2 = np.zeros(NN)
        result = np.zeros((NN, N))
        for i in range(0, NN):
            chain = self.map_clip(chain, ad, bias, a, b,
                                  noise_strength)  # calls Euler integration step, chose the map_xxx function to select different nonlinearities
            result[i, :] = chain
            ens2[i] = self.ising_energy(chain, ad, bias)

        return [0.5 * (np.sign(result[:, :n_hid]) + 1), 0.5 * (np.sign(result[:, n_hid:]) + 1)]
    def gen_minst_image(self,X):
        "生成图片函数"
        plt.rcParams['image.cmap'] = 'gray'
        return np.rollaxis(np.rollaxis(X[0:200].reshape(20,-1,8,8),0,2),1,3).reshape(-1,20*8)
    def transform(self, X):
        "检查模型是否拟合+检查数据的类型和格式"
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        return self._mean_hiddens(X)
    def _mean_hiddens(self, v):
        p = safe_sparse_dot(v, self.components_.T)
        p += self.intercept_hidden_
        return expit(p, out=p)
    def _sample_hiddens(self, v, rng):
        p = self._mean_hiddens(v)
        return (rng.random_sample(size=p.shape) < p)
    def _sample_visibles(self, h, rng):
        p = np.dot(h, self.components_)
        p += self.intercept_visible_
        expit(p, out=p)
        return (rng.random_sample(size=p.shape) < p)
    def _free_energy(self, v):
        "用于计算可见层状态的自由能"
        return (- safe_sparse_dot(v, self.intercept_visible_)
                - np.logaddexp(0, safe_sparse_dot(v, self.components_.T)
                               + self.intercept_hidden_).sum(axis=1))
    def gibbs(self, v):
        "用于从模型中生成新的样本"
        check_is_fitted(self)
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        h_ = self._sample_hiddens(v, self.random_state_)
        v_ = self._sample_visibles(h_, self.random_state_)
        return v_

    def _fit(self, v_pos, rng):
        "基于对比散度的模型训练方法"
        h_pos = self._mean_hiddens(v_pos)
        self.make_jij()
        h_neg, v_neg = self.ising_sample(1000, len(self.components_))
        lengg = (len(h_neg))
        lr = float(self.learning_rate) / v_pos.shape[0]
        update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
        update -= np.dot(h_neg.T, v_neg) / (lengg / self.batch_size)
        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0) / (lengg / self.batch_size))
        self.intercept_visible_ += lr * (np.asarray(
            v_pos.sum(axis=0)).squeeze() -
                                         v_neg.sum(axis=0) / (lengg / self.batch_size))

        h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
        self.h_samples_ = np.floor(h_neg, h_neg)
    def score_samples(self, X):
        "随机翻转每个样本的一个特征生成对比样本进行评分"
        check_is_fitted(self)
        v = check_array(X, accept_sparse='csr')
        rng = check_random_state(self.random_state)
        # Randomly corrupt one feature in each sample in v.
        ind = (np.arange(v.shape[0]),
               rng.randint(0, v.shape[1], v.shape[0]))
        if sp.issparse(v):
            data = -2 * v[ind] + 1
            v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
        else:
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]
        fe = self._free_energy(v)
        fe_ = self._free_energy(v_)
        return v.shape[1] * log_logistic(fe_ - fe)
    def make_jij(self):
        "主要功能是生成一个矩阵，该矩阵包含了隐藏层和可见层之间的权重信息以及偏置项的组合值，并将结果保存到文件中"
        temp = np.zeros(0)
        n_hidd = len(self.components_)
        n_vis = len(self.components_[0])
        for i in range(0, n_hidd):
            for j in range(0, n_vis):
                if (self.components_[i, j] != 0):
                    temp = np.append(temp, [(i + 1), (j + 1 + n_hidd), 0.25 * (self.components_[i, j])])
                    temp = np.append(temp, [(j + 1 + n_hidd), (i + 1), 0.25 * (self.components_[i, j])])
        for i in range(0, n_hidd):
            temp2 = 0
            for j in range(0, n_vis):
                temp2 += self.components_[i, j]
            temp = np.append(temp, [i + 1, i + 1, (0.5 * self.intercept_hidden_[i] + 0.25 * temp2)])
        for i in range(0, n_vis):
            temp2 = 0
            for j in range(0, n_hidd):
                temp2 += self.components_[j, i]
            temp = np.append(temp,
                             [n_hidd + i + 1, n_hidd + i + 1, (0.5 * self.intercept_visible_[i] + 0.25 * temp2)])

        temp = temp.reshape(2 * n_hidd * n_vis + n_hidd + n_vis, 3)
        np.savetxt(r'bla.txt', temp, delimiter=' ', newline='\n', fmt='%.10f')

    def fit(self, X, Y_train, X_test, Y_test):
        "训练 测试函数"
        logistic = linear_model.LogisticRegression(solver='newton-cg', tol=1)
        logistic.C = 6000.0
        X = self._validate_data(X, accept_sparse='csr', dtype=np.float64)
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)

        self.batch_size = 100
        self.components_ = np.asarray(
            rng.normal(0, 0.01, (self.n_components, X.shape[1])),
            order='F')
        self.intercept_hidden_ = np.zeros(self.n_components, )
        self.intercept_visible_ = np.zeros(X.shape[1], )
        self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))  # 7188/100 71
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples=n_samples))
        verbose = self.verbose
        begin = time.time()
        numnum = 0  # n_batch * self.n_iter
        likelhd = []
        predic = []
        for iteration in range(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                numnum += 1
                self._fit(X[batch_slice], rng)
                if verbose:
                    end = time.time()
                    likelhd.append(self.score_samples(X).mean())
                    print("[%s] Iteration %d, pseudo-likelihood = %.2f," " time = %.2fs" % (
                    type(self).__name__, numnum, self.score_samples(X).mean(), end - begin))
                    begin = end
                    if self.model_name == "LogisticRegression":
                        rf = linear_model.LogisticRegression(solver='newton-cg', tol=1)
                        rf.C = 6000.0
                    elif self.model_name == "LDA":
                        rf = LinearDiscriminantAnalysis()
                    elif self.model_name == "SVM":
                        rf = SVC(kernel='linear', C=1.0, gamma='scale', decision_function_shape='ovr')
                    elif self.model_name == "MLPClassifier":
                        rf = MLPClassifier(hidden_layer_sizes=(5, 10), solver='adam', max_iter=1000,
                                           random_state=42)
                    elif self.model_name == "GaussianNB":
                        rf = GaussianNB()
                    elif self.model_name == "MultinomialNB":
                        rf = MultinomialNB()
                    elif self.model_name == "DecisionTreeClassifier":
                        rf = DecisionTreeClassifier(random_state=42)
                    elif self.model_name == "Perctron":
                        rf = OneVsRestClassifier(Perceptron(random_state=42))
                    predic.append(
                        rf.fit((self._mean_hiddens(X)), Y_train).score((self._mean_hiddens(X_test)), Y_test))
                    print(predic[numnum - 1])
                    if (numnum - 1) % 20 == 0:
                        xx = X[:40].copy()
                        for ii in range(1000):
                            for n in range(40):
                                xx[n] = self.gibbs(xx[n])
                        plt.figure(figsize=(16, 2))
                        plt.imshow(self.gen_minst_image(xx))
                        plt.show()
        plt.xlabel('Training iteration')  # X轴标签
        plt.ylabel('Pseudolikelihood ')  # Y轴标签
        plt.plot(likelhd)
        plt.show()
        plt.plot(predic)
        plt.xlabel('Training iteration')  # X轴标签
        plt.ylabel('Prediction accuracy')  # Y轴标签
        plt.show()
        filename1 = f"out_likelhd_{self.model_name}.txt"
        np.savetxt(filename1, likelhd)
        filename2 = f"out_predict_{self.model_name}.txt"
        np.savetxt(filename2, predic)
        return self
    def train_model(self,X_train, y_train, model_name):
        "选择你的模型"
        if model_name == "LogisticRegression":
            model = linear_model.LogisticRegression(solver='newton-cg', tol=1)
            model.C = 6000.0
        elif model_name == "LDA":
            model = LinearDiscriminantAnalysis()
        elif model_name == "SVM":
            model = SVC(kernel='linear', C=1.0, gamma='scale', decision_function_shape='ovr')
        elif model_name == "MLPClassifier":
            model = MLPClassifier(hidden_layer_sizes=(5, 10), solver='adam', max_iter=1000, random_state=42)
        elif model_name == "GaussianNB":
            model = GaussianNB()
        elif model_name == "MultinomialNB":
            model = MultinomialNB()
        elif model_name == "DecisionTreeClassifier":
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == "Perctron":
            model = OneVsRestClassifier(Perceptron(random_state=42))
        model.fit(X_train, y_train)
        return model
    def evaluate_model(self,model, X_test, y_test):
        "预测并评估模型性能"
        y_pred = model.predict(X_test)
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        # 打印分类报告
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    def visualize_predictions(self,model, X_test, y_test, n_samples=5):
        "可视化部分预测结果"
        # 获取预测结果
        y_pred = model.predict(X_test)
        # 随机选择 n_samples 个样本进行可视化
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        plt.figure(figsize=(10, 6))
        for i, idx in enumerate(indices, start=1):
            plt.subplot(1, n_samples, i)
            img = X_test[idx].reshape(8, 8)
            plt.imshow(img, cmap='gray')
            plt.title(f'True: {y_test[idx]}\nPred: {y_pred[idx]}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    def translate_image(self,image, direction):
        "图片转换"
        if direction == 'up':
            return shift(image, [-1, 0], mode='constant', cval=0)
        elif direction == 'down':
            return shift(image, [1, 0], mode='constant', cval=0)
        elif direction == 'left':
            return shift(image, [0, -1], mode='constant', cval=0)
        elif direction == 'right':
            return shift(image, [0, 1], mode='constant', cval=0)
        else:
            raise ValueError("Invalid direction. Use 'up', 'down', 'left', or 'right'.")
    def load_data(self):
        "载入图片数据"
        digits = load_digits()
        # 获取图像数据和标签
        images = digits.images  # 8x8 的图像矩阵
        plt.imshow(images[2], origin='lower', cmap="gray")
        labels = digits.target  # 对应的标签
        # 扩展数据集
        expanded_images = []
        expanded_labels = []
        for image, label in zip(images, labels):
            # 原始图像
            expanded_images.append(image)
            expanded_labels.append(label)
            # 向四个方向平移
            for direction in ['up', 'down', 'left', 'right']:
                translated_image = self.translate_image(image, direction)
                expanded_images.append(translated_image)
                expanded_labels.append(label)
                # 将列表转换为 NumPy 数组
        expanded_images = np.array(expanded_images)
        expanded_labels = np.array(expanded_labels)
        # 将图像数据展平为二维数组 (n_samples, 64)
        n_samples = expanded_images.shape[0]
        data = expanded_images.reshape((n_samples, -1))
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(data, expanded_labels, test_size=0.2, random_state=42)
        X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)
        X_test = (X_test - np.min(X_test, 0)) / (np.max(X_test, 0) + 0.0001)
        return X_train, X_test, y_train, y_test