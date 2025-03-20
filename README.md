RBM_verify 文件夹包含复现的notebook文件和训练时的似然+预测值数据 采用线性回归

RBM_verify_randomforest 文件夹包含复现的notebook文件和训练时的似然+预测值数据 采用随机森林

out_pred.txt 是RBM复现图像分类时的预测输出值

out_likelhd.txt 是RBM复现图像分类时的似然输出值

cim_verify.ipynb 是小规模Spin变量上的玻尔兹曼采样验证notebook文件

小规模Spin变量进行采样的模型如下所示

<img width="120" alt="图片" src="https://github.com/user-attachments/assets/834c79f1-8afb-4e61-b7e8-4e2cd73e1314" />


cim vs mcmc包含了cim采样和mcmc采样对比的代码，其中mcmc温度T和噪声noise方差存在对应关系，需要手动调节才能让他们两者比较相符。verify3.ipynb自己生成数据进行验证，cim_verify2.ipynb读取给定文件数据进行验证

