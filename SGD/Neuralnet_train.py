import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
from dataset.mnist import load_mnist
from SGD.TwoLayerNet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist\
    (normalize=False,flatten=True,one_hot_label=True)
train_loss = []
'''超参数'''
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size = 784, hide_size = 50, output_size = 10)
for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch) # 高速版!

    # 更新参数
    for key in ('w1', 'b1', 'w2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss.append(loss)
    print(train_loss)
output = open('network_params.pkl','wb')
pickle.dump(network.params,output)
output.close()
