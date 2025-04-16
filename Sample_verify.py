"""
本代码包含在大规模方格结点上进行采样的类
"""
import numpy as np
from collections import Counter
import pandas as pd
import random
class Sample:
    def __init__(self,noise,T,N):
        self.noise=noise #噪声均方差
        self.T=T #mcmc温度
        self.N=N #采样次数
    def map_clip(self,old, coup, bias, a, b, noi):
        "linear map with clipping"
        return np.clip(a * old + b * np.dot(coup, old) + .40 * b * bias + np.random.normal(0, noi, len(old)), -.4, .4)
    def ising_energy(self,stuff, ad, bias):
        "计算ising能量"
        return -0.5 * 1 * np.dot(stuff, np.dot(ad, stuff)) - np.sum(np.dot(bias, stuff))
    def ising_sample(self,array,  _a, _b):
        "快速采样"
        global result
        global ens2
        global ad
        global chain
        global bias
        NN=self.N
        noise=self.noise
        test = np.array(array)
        N = int(np.max(test))  # number of nodes
        ad = np.zeros((N, N))  # adjacency matrix
        for i in range(0, len(test)):  # initialize
            ad[int(test[i, 0]) - 1, int(test[i, 1]) - 1] += (test[i, 2])
        bias = np.zeros(N)
        for i in range(0, N):
            bias[i] += ad[i, i]
            ad[i, i] = 0
        # parameters for alpha and beta
        a = _a
        b = _b
        noise_strength = noise
        chain = np.zeros(N)
        ens2 = np.zeros(NN)
        result = []
        for i in range(0, NN):
            chain = self.map_clip(chain, ad, bias, a, b, noise_strength)
            chain_a = np.sign(chain)
            ens2[i] = self.ising_energy(chain_a, ad, bias)
            result.append(ens2[i])
        result = np.array(result)
        return result
    def calculate(self,array):
        "计算数值和频次的函数"
        counter = Counter(array)
        numbers = np.array(list(counter.keys()))  # 数的大小
        frequencies = np.array(list(counter.values()))  # 频率
        all_counts = np.sum(frequencies)
        P = frequencies / all_counts
        # 将 numbers 和 P 配对
        paired = list(zip(numbers, P))
        # 按 numbers 的值进行排序（默认升序）
        paired_sorted = sorted(paired, key=lambda x: x[0])
        # 解压排序后的结果
        numbers, P = zip(*paired_sorted)
        return np.array(numbers), np.array(P)
    def generate_data(self,N):
        "生成数据矩阵并且保存  数据的图结点每个节点和上下左右四个节点连接"
        # 设置随机数种子以确保结果可重复
        random.seed(42)  # 可以选择任意整数作为种子值
        # 初始化边列表
        edges = []
        # 添加水平边
        rows = N
        cols = N  ###结点数为N*N
        random_num = -1
        for row in range(rows):
            for col in range(cols - 1):  # 每行最多有 cols-1 条水平边
                node1 = row * cols + col + 1  # 当前节点编号
                node2 = node1 + 1  # 右侧相邻节点编号
                edges.append([node1, node2, random_num])
                edges.append([node2, node1, random_num])
        # 添加垂直边
        for col in range(cols):
            for row in range(rows - 1):  # 每列最多有 rows-1 条垂直边
                node1 = row * cols + col + 1  # 当前节点编号
                node2 = node1 + cols  # 下方相邻节点编号
                edges.append([node1, node2, random_num])
                edges.append([node2, node1, random_num])
        # 转换为 DataFrame 并写入 txt 文件
        matrix=np.array(edges)
        df = pd.DataFrame(matrix)
        df.to_csv('matrix.txt', header=False, index=False, sep=' ')  # 不写入表头和索引，用逗号分隔
        print("矩阵已成功写入 matrix.txt 文件")
        return np.array(edges)

    def flip2(self,e_new, e_old, new, old):
        "mcmc采样的子函数"
        k=1
        T=self.T
        flip_prob = (np.exp(-(e_new - e_old) / (k * T)))
        if (np.random.rand() < flip_prob):
            return new
        else:
            return old

    def flip(self,stuff, index, ad, bias):
        "mcmc采样反转函数"
        temp = np.copy(stuff)
        temp[index] = -1 * temp[index]
        temp_energy = self.ising_energy(temp, ad, bias)
        energy = self.ising_energy(stuff, ad, bias)
        global ens
        # ens=np.append(ens,energy)
        if (temp_energy <= energy):
            return temp
        else:
            return self.flip2(temp_energy, energy, temp, stuff)
    def mcmc_sample(self,NN, array):
        "mcmc采样"
        test = array  # edge list
        N = int(np.max(test))  # number of nodes
        ad = np.zeros((N, N))  # adjacency matrix
        for i in range(0, len(test)):  # initialize
            ad[int(test[i, 0]) - 1, int(test[i, 1]) - 1] += (test[i, 2])
        bias = np.zeros((N, N))
        for i in range(0, N):
            bias[i, i] += ad[i, i]
            ad[i, i] = 0
        chain = np.ones(N)  ##np.sign(np.random.rand(N)*2+0.01-1)
        result = []
        for i in range(0, NN):
            # chain=flip(chain,int(np.random.rand()*N))    #random update order
            chain = self.flip(chain, i % N, ad, bias)  # typewirter update order, does not work well in 1D, better in 2D
            if (i > 1000 and i % 1 == 0):
                result.append(self.ising_energy(chain, ad, bias))
        return np.array(result)