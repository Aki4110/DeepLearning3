import numpy as np

# クラスVariableの定義
class Variable:

    #クラスのコンストラクタ
    def __init__(self, data):
        self.data = data #インスタンス変数

# クラスFunctionの定義
class Function:

    # pythonの特殊メソッド  Variableからのデータを取り出す。
    def __call__(self, input):
        x = input.data
        y = x**2
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x**2

x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)