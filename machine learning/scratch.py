import numpy as np
import random

class LearningAssetsModule:

    # Mathematical functions
    def __init__(self, name):
        self.name = name

    def ReLU(self, z):
        print(z.shape)
        index_greater = np.nonzero(z<0)
        for index_pair in list(zip(index_greater[0], index_greater[1])):
            z[index_pair[0], index_greater[1]] = 0
        return z

    def MSE_with_error(self, y_actual, y_forcasted, node_error = True):
        n = y_forcasted.size
        MSE = 0
        if node_error:
            e_l= np.zeros((n, 1))
        for index, y_i_pair in enumerate(list(zip(y_actual, y_forcasted))):
            MSE = MSE + (y_i_pair[0]-y_i_pair[1])**2/n
            if node_error:
                if y_i_pair[1]>0:
                    e_l[index, 0] = 2 * (y_i_pair[0]-y_i_pair[1])/n
                else:
                    e_l[index, 0] = 0
        if node_error:
            return MSE, e_l
        else:
            return MSE


    def forward_pass(self, x, W, b):
        z = np.matmul(W, x) + b
        a = self.ReLU(z)
        return a

    def CalculateError(self, W, e_l2, a_l1):
        index_greater = np.nonzero(a_l1 != 0)
        for index_pair in list(zip(index_greater)):
            a_l1[index_pair[0], index_greater[1]] = 1
        e_l1 = np.matmul(W.T, e_l2)*a_l1

        return e_l1

    def CostvsBias_grad(self, e_l1):
        return e_l1

    def CostvsWeight_grad(self, e_l2, a_l1):
        return np.matmul(e_l2, a_l1.T)



class MyModel(LearningAssetsModule):

    def __init__(self, name, input_layer_n, layer_shape):
        W_list = []
        b_list = []
        for index, n in enumerate(layer_shape):
            W_list = W_list + [np.random.rand(n*input_layer_n).reshape(n, input_layer_n)]
            b_list = b_list + [(np.random.rand(n)).reshape(n, 1)]
            input_layer_n = n

        self.W_list = W_list
        self.b_list = b_list
        super().__init__(name)

    def Start_Learning(self, x_l0, y_forcasted, lr):

        ### LIST OF ACTIVATIONS
        a = [x_l0]
        W_list = self.W_list.copy()
        b_list = self.b_list.copy()
        for W, b in zip(W_list, b_list):
            temp = self.forward_pass(x_l0, W, b)
            a = a + [temp]
            x_l0 = temp
        ### ERROR CALCULATION

        MSE, e_lastl = self.MSE_with_error(a[-1], y_forcasted)

        print(f'Cost is: {MSE}')
        a_reversed = a[1:-1]
        W_reversed = W_list[1:]
        a_reversed.reverse()
        W_reversed.reverse()

        e = [e_lastl]
        for index, weight_activation_pair in enumerate(zip(W_reversed, a_reversed)):
            e = e + [self.CalculateError(weight_activation_pair[0], e[index], weight_activation_pair[1])]

        ### BACKPROPAGATION AND GRADIENT DESCENT

        CostvsBias = []
        CostvsWeight = []

        a_reversed_2 = a[0:-1]
        a_reversed_2.reverse()
        for a_l, e_l2 in zip(a_reversed_2, e):
            CostvsBias = CostvsBias + [self.CostvsBias_grad(e_l2)]
            CostvsWeight = CostvsWeight + [self.CostvsWeight_grad(e_l2, a_l)]

        CostvsWeight.reverse()
        CostvsBias.reverse()

        for index, layer_bias_weight_grad in enumerate(zip(b_list, W_list, CostvsBias, CostvsWeight)):
            self.b_list[index] = layer_bias_weight_grad[0] - (lr * layer_bias_weight_grad[2])
            self.W_list[index] = layer_bias_weight_grad[1] - (lr * layer_bias_weight_grad[3])


    def Predict(self, x_l0):
        a = [x_l0]
        W_list = self.W_list
        b_list = self.b_list
        for W, b in zip(W_list, b_list):
            temp = self.forward_pass(x_l0, W, b)
            a = a + [temp]
            x_l0 = temp

        return a[-1]

x = np.random.randint(1, 101, 100)
y_forcasted = 4*x+5
input_output_pairs = list(zip(x, y_forcasted))

random.shuffle(input_output_pairs)

train_pairs = input_output_pairs[0:15]
test_pairs = input_output_pairs[15:21]


Model_1 = MyModel('Model_1', 1, [2, 2, 1])


epoch_count = 1
for train_pair in train_pairs:
    print(f"Epoch Count: {epoch_count}")
    epoch_count+=1
    # print(train_pair)
    Model_1.Start_Learning(np.array(train_pair[0]).reshape(1, 1), np.array(train_pair[1]).reshape(1, 1), lr=0.001)


for test_pair in test_pairs:

    print(test_pair)
    y_prediction = Model_1.Predict(np.array(test_pair[0]).reshape(1, 1))

    print(f"Prediction: {y_prediction} | Forcasted: {test_pair[1]} | Input: {test_pair[0]}")
