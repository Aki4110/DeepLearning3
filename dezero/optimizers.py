import numpy as np

# 損失関数を最小化するために使用されるアルゴリズム
class Optimizer:
    # クラスのコンストラクタ
    def __init__(self):
        # 2つのインスタンス変数の初期化
        self.target = None
        self.hooks = [] # フックス

    def setup(self, target):
        self.target = target
        return self
    
    def update(self):
        # None以外のパラメータをリストにまとめる
        params = [p for p in self.target.params() if p.grad is not None]

        # 前処理オプション
        for f in self.hooks:
            f(params)

        # パラメータの更新
        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError #プログラムの実行中に例外を手動で発生させるために使用されます
    
    def add_hook(self, f):
        self.hooks.append(f)

# 確率的勾配降下法（Stochastic Gradient Descent）
class SGD(Optimizer):

    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data

class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v