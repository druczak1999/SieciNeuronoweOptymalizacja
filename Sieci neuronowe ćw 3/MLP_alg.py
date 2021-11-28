import random
INFINITY = float('inf')
WEIGHTS =[]
BIAS = []
MOMENTUM=[]
CACHE=[]
from MLP import MLP
import numpy as np

class MLP_alogorythm:
   
    def algorythm(self,X,Y,number,WEIGHTS,BIAS,fun1,fun2, epochs, prog, mi, MOMENTUM, CACHE):
        mlp = MLP()
       
        a1, sum_for_all_1 = mlp.hidden_layer(X,number,WEIGHTS[0],BIAS[0],fun1, 1)
        y, a_for_all_neurons = mlp.output_layer(a1,10,WEIGHTS[2],BIAS[2])  
       
        diffs,howmuchgood = mlp.diff_output(y,Y,a_for_all_neurons)
        diffs_h1 = mlp.diff_hiden(WEIGHTS[0],diffs,BIAS[0],fun1,sum_for_all_1)
        
        #for adam
        # WEIGHTS[2], MOMENTUM[2],CACHE[2], BIAS[2] = mlp.weights_update_adam(WEIGHTS[2],mi,y,diffs, MOMENTUM[2], CACHE[2], BIAS[2])
        # WEIGHTS[0], MOMENTUM[0],CACHE[0], BIAS[0] = mlp.weights_update_adam(WEIGHTS[0],mi,a1,diffs_h1,MOMENTUM[0],CACHE[0],BIAS[0])
        
        #for rest
        WEIGHTS[2], MOMENTUM[2], BIAS[2] = mlp.weights_update_adadelta(WEIGHTS[2],mi,y,diffs,0.7, MOMENTUM[2],BIAS[2])
        WEIGHTS[0], MOMENTUM[0], BIAS[0] = mlp.weights_update_adadelta(WEIGHTS[0],mi,a1,diffs_h1,0.7,MOMENTUM[0], BIAS[0])
        
        diffs_pow=[]
        for i in range(len(diffs)):
            diffs_pow.append(diffs[i]**2)    
        prog = sum(diffs_pow)/2
        epochs+=1

        return WEIGHTS, BIAS, epochs, prog, MOMENTUM, CACHE, howmuchgood

    def input_layer_batch(self,batch,i,train_X, train_y):
        return train_X[i*batch:batch+i*batch], train_y[i*batch:batch+i*batch]

    def read_mnist_batch(self,len):
        mlp = MLP()
        train_X, train_y, test_X, test_y = mlp.mnist()
        return train_X, train_y, test_X, test_y

    def learn_algorythm_batch(self, number, whmin, whmax, mi, threshold, batch, len_of_input):
        mlp = MLP()
        prog=INFINITY
        WEIGHTS=[]
        BIAS=[]
        MOMENTUM=[]
        CACHE=[]
        CACHE.append(np.zeros(shape=(number, batch)))
        CACHE.append(np.zeros(shape=((int)(number*0.7),batch)))
        CACHE.append(np.zeros(shape=(10, batch)))
        prog = INFINITY
        epochs = 0
        ep=0
        fun1 = "relu"
        fun2 = "sigma"
        WEIGHTS,BIAS, MOMENTUM  = mlp.set_weights_and_bias(number, whmin, whmax, batch, WEIGHTS, BIAS, MOMENTUM)
        while prog > threshold:
            train_X, train_y, test_X, test_y = mlp.mnist();
            rng_state = random.getstate()
            random.shuffle(train_X)
            random.setstate(rng_state)
            random.shuffle(train_y)
            progs=[]
            howgood=[]
            for i in range(int(len_of_input/batch)):
                X, Y = self.input_layer_batch(batch,i, train_X, train_y)
                # print("weights")
                WEIGHTS,BIAS,epochs,prog1, MOMENTUM, CACHE,howmuchgood = self.algorythm(X, Y, number, WEIGHTS, BIAS, fun1, fun2, epochs, prog, mi, MOMENTUM, CACHE)
                progs.append(prog1)
                howgood.append(howmuchgood)
            prog = sum(progs)/len(progs)
            print("PROG PO CALYM",prog)
            ep+=1
        print(epochs)
        return ep,sum(howgood)/len(howgood)