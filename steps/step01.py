import numpy as np
import unittest

# クラスVariableの定義
# Variableクラスを使用すると、逆伝播の手続きを自動的に処理できます。
class Variable:

    #クラスのコンストラクタ
    def __init__(self, data):

        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data #インスタンス変数
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    """pop
    my_list = [1, 2, 3, 4, 5]

    # pop() メソッドを使用してリストから最後の要素を取り出す
    last_element = my_list.pop()

    print(last_element)  # 最後の要素（5）が表示される
    print(my_list)  # 最後の要素がリストから削除されたので、[1, 2, 3, 4] が表示される

    """
    def backward(self):

        if self.grad is None:
            self.grad = np.ones_like(self.data) # np.ones_like(arr) 関数は、引数として渡されたNumPy配列 arr と同じ形状を持つ新しいNumPy配列を作成します。新しい配列は、全ての要素が1で初期化されます。具体的な使用例を示します：
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

# クラスFunctionの定義
class Function:

    # pythonの特殊メソッド  Variableからのデータを取り出す。
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self) # 出力変数に生みの親を覚えさせる
        self.input = input # 入力された変数を覚える
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

# Functionクラスを継承
# Functionクラスの属性とメソッドをSquareクラスで使用できるようになります

class Square(Function):
    def forward(self, x):
        return x**2
    
    def backward(self, gy):
        x = self.input.data # self.input は Function クラスのメソッド __call__ 内で保存された入力変数を指します。self.input.data は入力変数のデータを表します。
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

class SquareTest(unittest.TestCase):

    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEquals(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerial_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
        
def numerial_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
# y.grad のように記述できるのは、Variable クラス内で grad というインスタンス変数が定義されているからです。
# B クラスの __call__ メソッド内で、b は新しい Variable インスタンスとして生成され、self.input に a が設定されています。これにより、b は B クラスの __call__ メソッド内で入力変数 a と関連づけられます。
"""y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)"""

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def main():
    x = Variable(np.array(0.5))
    a = square(x)
    b = exp(a)
    y = square(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

if __name__ == "__main__":
    main()