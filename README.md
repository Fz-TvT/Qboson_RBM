RBM_8*8.ipynb 是RBM复现图像分类时的notebook文件

out_pred.txt 是RBM复现图像分类时的预测输出值

out_likelhd.txt 是RBM复现图像分类时的似然输出值

cim_verify.ipynb 是小规模Spin变量上的玻尔兹曼采样验证notebook文件

小规模Spin变量进行采样的模型如下所示

![图片](https://github.com/user-attachments/assets/b58a6319-7e89-4f70-bd18-1a50225d5148) 

cim vs mcmc包含了cim采样和mcmc采样对比的代码，其中mcmc温度T和噪声noise方差存在对应关系，需要手动调节才能让他们两者比较相符。
