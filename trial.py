from Tanitra import Tanitra
from Parata import PraveshParata, GuptaParata, NirgamParata, Samasuchaka,ConvLayer2D,MaxPoolingLayer2D
from Pratirup import AnukramikPratirup
import cupy as cp
from tensorflow.keras.datasets import mnist

(x_train_ref, y_train), (x_test_ref, y_test) = mnist.load_data()
x_train_ref, x_test_ref = x_train_ref / 255.0, x_test_ref / 255.0

x_train = cp.zeros((300, 28,28))
x_test = cp.zeros((50,784))
y_train_revised = cp.zeros((300 ,10))
y_test_revised = cp.zeros((50,10))

for i in range(len(x_train[:300])):
    x_train[i] = cp.array(x_train_ref[i])

for i in range(len(x_train[:50])):
    x_train[i] = cp.array(x_test_ref[i])

for i in range(len(y_train[:300])):
    y_train_revised[i][cp.array(y_train)[i]] = 1

for i in range(len(y_test[:50])):
    y_test_revised[i][cp.array(y_test)[i]] = 1

print(y_test_revised,y_train_revised)


x_train = Tanitra(x_train)
x_test = Tanitra(x_test)
y_train_revised = Tanitra(y_train_revised)
y_test_revised = Tanitra(y_test_revised)

model = AnukramikPratirup()

normalizer = Samasuchaka('min-max')

model.add(PraveshParata((28,28)))
model.add(ConvLayer2D(2,5,6,'relu'))
model.add(MaxPoolingLayer2D(2,4))
model.add(GuptaParata(100, 'relu',))
model.add(NirgamParata(10, 'softmax',))

model.learn ( x_train, y_train_revised,epochs=2,tol = 0.000000001)
result = []
for i in x_test:
    result.append([model.estimate(i).data.argmax(),y_test[i].data.argmax()])
print(result)
