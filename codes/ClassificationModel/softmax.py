import torch
# import torchvision
import numpy as np
import sys
# sys.path.append("/home/kesci/input")
# import d2lzh1981 as d2l

print(torch.__version__)
# print(torchvision.__version__)


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    # print("X size is ", X_exp.size())
    # print("partition size is ", partition, partition.size())
    return X_exp / partition  # 这里应用了广播机制

X = torch.tensor([[10, 10.1, 10.2],[-2,-1,0]])
X_prob = softmax(X)
print(X_prob, '\n', X_prob.sum(dim=1))