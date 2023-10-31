if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ライブラリ全体をimport
import numpy as np
import matplotlib.pyplot as plt
import dezero.layers as L
import dezero.functions as F
import dezero.optimizers as op

# モジュールごとにimport
from dezero.models import MLP
from dezero.models import Model
from dezero.core import Variable, Function, as_variable

def softmaxld(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y

x = Variable(np.array([[0.2, -0.4]]))
y = model(x)
p = softmaxld(y)
print(y)
print(p)