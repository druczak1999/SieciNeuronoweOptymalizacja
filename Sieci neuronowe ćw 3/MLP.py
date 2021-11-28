import numpy as np
import random
from keras.datasets import mnist
from MLP_active_functions import Active_functions

class MLP:

    def mnist(self):
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        return train_X, train_y, test_X, test_y

    def input_layer(self):
        train_X, train_y, test_X, test_y = self.mnist()
        validate_X = train_X[100:200]
        validate_y = train_y[100:200]
        return train_X[:100], train_y[:100], test_X[:10], test_y[:10], validate_X, validate_y
        
    def hidden_layer(self, x, number, weights_for_all_neurons, bias_for_all_neurons, fun, which_layer):
        af = Active_functions()
        sum_for_all = []
        a_for_all_neurons = []
        for c in range(len(x)): #dla kazdego obrazka
            a=[]
            sums=[]
            for b in range(number): #dla kazdego neuronu
                sum= self.sum_all(x[c], weights_for_all_neurons[b][c], bias_for_all_neurons[b], which_layer)
                sums.append(sum)
                a.append(af.choose_fun(fun, sum))
            a_for_all_neurons.append(a) 
            sum_for_all.append(sums)
        return a_for_all_neurons, sum_for_all

    def output_layer(self, x, number, weights_for_all_neurons, bias_for_all_neurons):
        af = Active_functions()
        a_for_all_neurons = []
        for c in range(len(x)): #dla kazdego obrazka
            a=[]
            for b in range(number): #dla kazdego neuronu
                sum = self.sum_all(x[c], weights_for_all_neurons[b][c], bias_for_all_neurons[b],2)
                a.append(sum)
            a_for_all_neurons.append(a)

        y=[]
        for i in range(len(a_for_all_neurons)): y.append(af.softmax(a_for_all_neurons[i]))

        return y, a_for_all_neurons

    def diff_hiden(self,weights,diffs,bias,fun,sums):
        diffs_h=[]
        af = Active_functions()
        for i in range (len(sums)):
            for j in range(len(sums[i])):
                z = af.choose_deriv(sums[i][j],fun)
                sum=(diffs[i] * weights[j][i]) + bias[j]
            diffs_h.append(z*sum)
        return diffs_h

    def diff_output(self,y_predict, labels, sums):
        af = Active_functions()
        diffs=[]
        howmuchgood=0
        for i in range(len(y_predict)):
            max_val = max(y_predict[i])
            diff = labels[i] - y_predict[i].tolist().index(max_val)
            if diff==0: howmuchgood+=1
            diff_soft = diff * af.softmax_max(sums[i])
            diffs.append(diff_soft)
        return diffs, howmuchgood/len(y_predict)

    def weights_update(self,weights,mi,a,diff,c,b,bias):
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights[i][j] += mi * a[j][i] * diff[j]
            bias[i] += mi * (sum(diff)/len(diff))
        return weights,c,bias

    def weights_update_momentum_sgd(self,weights,mi,a,diff,change, momentum, bias):
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                momentum[i][j] = momentum[i][j]*change - a[j][i]*mi*diff[j]
                weights[i][j] += momentum[i][j]
            bias[i] +=sum(momentum[i])/len(momentum[i])
        return weights, momentum, bias

    def weights_update_momentum_nesterov(self,weights,mi,a,diff,change, momentum, bias):
        for i in range(len(weights)):
            temp_bias=[]
            for j in range(len(weights[i])):
                temp_mom = momentum[i][j]
                temp_bias.append(temp_mom)
                momentum[i][j] = momentum[i][j]*change - a[j][i]*mi*diff[j]
                weights[i][j] += -1*change*temp_mom + ((1+change)*momentum[i][j])
            bias[i]+= -1*change*(sum(temp_bias)/len(temp_bias)) + ((1+change)*(sum(momentum[i])/len(momentum[i])))
        return weights, momentum, bias

    def weights_update_adagrad(self,weights,mi,a,diff,change, gradient, bias):
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                gradient[i][j]=((a[j][i]*diff[j])**2)
                weights[i][j] += (-1*mi* a[j][i]*diff[j]) / (np.sqrt(gradient[i][j]) + np.finfo(np.float32).eps)
            bias[i]+= (-1*mi* (sum(a[i])/len(a[i]))*(sum(diff)/len(diff))) / (np.sqrt(sum(gradient[i])/len(gradient[i])) + np.finfo(np.float32).eps)
        return weights, gradient, bias

    def weights_update_adadelta(self,weights,mi,a,diff,change,gradient, bias):
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                gradient[i][j]= (change * gradient[i][j]) + ((1-change) * (a[j][i]*diff[j])**2)
                weights[i][j] += (-1 * a[j][i]*mi*diff[j]) / (np.sqrt(gradient[i][j]) + np.finfo(np.float32).eps)
            bias[i]+= (-1*mi* (sum(a[i])/len(a[i]))*(sum(diff)/len(diff))) / (np.sqrt(sum(gradient[i])/len(gradient[i])) + np.finfo(np.float32).eps)
        return weights, gradient, bias

    def weights_update_adam(self,weights,mi,a,diff,f_grad, p_grad, bias):
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                p_grad[i][j] =(0.9 * p_grad[i][j]) + (0.1*a[j][i]*diff[j])
                mt = p_grad[i][j]/(0.1**j)
                f_grad[i][j]=(0.999 * f_grad[i][j]) + (0.001 * ((a[j][i]*diff[j])**2))
                ct = f_grad[i][j]/(0.001**j)
                weights[i][j] += (-1 *mi*mt )/ (np.sqrt(ct) + np.finfo(np.float32).eps)
            bias[i] += (-1 *mi*sum(p_grad[i])/len(p_grad[i])/0.1**i ) / (np.sqrt(sum(f_grad[i])/len(f_grad[i])/0.001**i) + np.finfo(np.float32).eps)
        return weights, f_grad, p_grad, bias

    def random_weights_and_bias(self, number, x, whmax, whmin, initialize_weights_method):
        weights_for_all_neurons =[]
        momentum_for_all_neurons =[]
        bias_for_all_neurons=self.random_bias(number)
        for _ in range (number):
            weights_for_neuron = []
            momentum_for_neuron = []
            for _ in range (x):
                if initialize_weights_method=="xavier":
                    weights_for_neuron.append(random.uniform(whmin, whmax)*np.sqrt((2/(number+x))))
                elif initialize_weights_method=="he":
                    weights_for_neuron.append(random.uniform(whmin, whmax)*np.sqrt((2/number)))
                else:
                    weights_for_neuron.append(random.uniform(whmin, whmax))
                momentum_for_neuron.append(0)
            weights_for_all_neurons.append(weights_for_neuron)
            momentum_for_all_neurons.append(momentum_for_neuron)
        return weights_for_all_neurons, bias_for_all_neurons, momentum_for_all_neurons

    def random_bias(self,number):
        bias_for_all_neurons=[]
        for _ in range(number):
            bias_for_all_neurons.append(random.uniform(-0.2, 0.2))
        return bias_for_all_neurons

    def set_weights_and_bias(self,number,whmin,whmax,X,WEIGHTS,BIAS, MOMENTUM):
        WFHL, BFHL, MFHL =self.random_weights_and_bias(number,X,whmax, whmin,"ar")
        WSHL, BSHL, MSHL =self.random_weights_and_bias((int)((number*0.7)),X,whmax, whmin, "a")
        WOHL, BOHL, MOHL =self.random_weights_and_bias(10,X,whmax, whmin, "a")
        WEIGHTS.append(WFHL)
        WEIGHTS.append(WSHL)
        WEIGHTS.append(WOHL)
        BIAS.append(BFHL)
        BIAS.append(BSHL)
        BIAS.append(BOHL)
        MOMENTUM.append(MFHL)
        MOMENTUM.append(MSHL)
        MOMENTUM.append(MOHL)
        return WEIGHTS,BIAS, MOMENTUM

    def sum_all(self, X, w, b, l):
        sum = 0
        if(l==1):
            for i in range(len(X)):
                for j in range(len(X[i])):
                    sum += X[i][j] * w + b
        else:
            for j in range(len(X)): sum +=(X[j] * w) + b
        return sum

