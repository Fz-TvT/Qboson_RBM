"""
本代码的类包含利用三种不同方式在小规模图上进行采样的内容
"""
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import math
class Sample_small:
    def __init__(self,_a,_b,_way,_NN,start,step,num_element):
        "初始化参数"
        self.a=_a
        self.b=_b
        self.way=_way
        self.array=None
        self.NN=_NN #采样次数
        self.start=start
        self.step=step
        self.num_element=num_element
    def map_tanh(self,old,coup,bias,a,b,noi):
        "linear map with clipping"
        return np.tanh(a*old+b*np.dot(coup,old)+np.abs(np.max(old))*b*bias+np.random.normal(0,noi,len(old)))
    def map_clip(self,old,coup,bias,a,b,noi):
        "linear map with clipping"
        return np.clip(a*old+b*np.dot(coup,old)+.40*b*bias+np.random.normal(0,noi,len(old)),-.4,.4)
    def map_pol(self,old,coup,bias,a,b,noi):
        "linear map with clipping"
        return np.clip((a)*old-old**3+b*np.dot(coup,old)+np.abs(np.max(old))*b*bias+np.random.normal(0,noi,len(old)),-.4,.4)
    def ising_energy(self,stuff,ad,bias):
        "计算ising能量"
        return -0.5*1*np.dot(stuff,np.dot(ad,stuff))-np.sum(np.dot(bias,stuff))
    def ising_sample(self,array,NN,noise,_a,_b):
        "快速采样"
        global result
        global ens2
        global ad
        global chain
        global bias
        test=np.array(array)
        N=int(np.max(test)) #number of nodes
        ad=np.zeros((N,N)) #adjacency matrix
        for i in range(0,len(test)): #initialize
            ad[int(test[i,0])-1,int(test[i,1])-1]+=(test[i,2])
        bias=np.zeros(N)
        for i in range(0,N):
            bias[i]+=ad[i,i]
            ad[i,i]=0
        # parameters for alpha and beta
        a=_a
        b=_b
        noise_strength=noise
        chain=np.zeros(N)
        ens2=np.zeros(NN)
        result=[]
        for i in range(0,NN):
            if self.way=="map_tanh":
                chain=self.map_tanh(chain,ad,bias,a,b,noise_strength)
            elif self.way=="map_pol":
                chain=self.map_pol(chain,ad,bias,a,b,noise_strength)
            elif self.way=="map_clip":
                chain=self.map_clip(chain,ad,bias,a,b,noise_strength)
            chain_a=np.sign(chain)
            ens2[i]=self.ising_energy(chain_a,ad,bias)
            result.append(ens2[i])
        # return [0.5*(np.sign(result[:,:n_hid])+1),0.5*(np.sign(result[:,n_hid:])+1)]
        result=np.array(result)
        return result
    def Plot_T_noise(self):
        "不同噪声方差和温度下的对应关系图表"
        start = self.start
        step = self.step
        num_elements = self.num_element
        # 计算结束值
        stop = start + step * (num_elements - 1)
        # 创建噪声向量
        noise_levels = np.arange(start, stop + step, step)
        self.array = [[1, 2, -0.4],
                 [1, 1, 0.4],
                 [1, 3, 0.4],
                 [2, 1, -0.4],
                 [3, 1, 0.4],
                 [2, 2, -0.4],
                 [3, 3, 0.1],
                 [1, 4, 0.4],
                 [4, 1, 0.4],
                 [4, 4, -0.4]
                 ]
        kl_ans = []
        T_ans = []
        print(self.way)
        for noise in noise_levels:
            data = self.ising_sample(self.array, self.NN, noise, self.a, self.b)
            # 统计每个数的频率
            counter = Counter(data)
            numbers = np.array(list(counter.keys()))  # 数的大小
            frequencies = np.array(list(counter.values()))  # 频率
            all_counts = np.sum(frequencies)
            P = frequencies / all_counts
            x = numbers.flatten()
            y = np.log(P.flatten())
            coefficients = np.polyfit(x, y, 1)
            slope = coefficients[0]
            intercept = coefficients[1]  # 截距
            Q = np.exp(slope * x + intercept)
            kl = 0
            T = -1 / (slope)
            T_ans.append(round(T, 2))
            # 计算KL散度
            for i in range(len(P)):
                if P[i] > 0 and Q[i] > 0:  # 避免除以零的情况
                    kl += P[i] * math.log(P[i] / Q[i])
            kl_ans.append(kl)
        plt.xlabel('Noise_variance', fontsize=14)
        plt.ylabel('KL_divergence', fontsize=14)
        plt.yscale('symlog', linthresh=0.1)  # linthresh 参数定义了线性部分的范围
        plt.grid(True, which="both", linestyle='-', linewidth=0.1)
        plt.plot(noise_levels, np.array(kl_ans), color='blue', label='kl随着噪声变化曲线', marker='o', markersize=5,
                 markerfacecolor='black', markeredgecolor='black')
        # # 使用zip函数将两个数组打包成元组列表
        plt.show()
        data = zip(noise_levels, T_ans)
        plt.xlabel('Noise_variance', fontsize=14)
        plt.ylabel('Temperature(J)', fontsize=14)
        plt.plot(noise_levels, T_ans)
        plt.show()
    def Plot_noise(self,noise):
        "画出给定噪声方差下log(p)和能量的对应关系图"
        data = self.ising_sample(self.array, self.NN, noise, self.a, self.b)
        # 统计每个数的频率
        counter = Counter(data)
        numbers = np.array(list(counter.keys()))  # 数的大小
        frequencies = np.array(list(counter.values()))  # 频率
        all_counts = np.sum(frequencies)
        P = frequencies / all_counts
        x = numbers.flatten()
        y = np.log(P.flatten())
        coefficients = np.polyfit(x, y, 1)
        slope = coefficients[0]
        intercept = coefficients[1]  # 截距
        plt.scatter(numbers, y, c='blue')
        plt.plot(x, slope * x + intercept, color='black', label='拟合直线')
        plt.xlabel('Energy', fontsize=14)
        plt.ylabel('Log_Probabilities', fontsize=14)