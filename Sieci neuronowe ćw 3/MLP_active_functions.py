import numpy as np
import math

class Active_functions:

    def sigma(self, sum):
        return 1 / (1 + np.exp(-1*sum))

    def hiperbola(self, sum):
        return np.tanh(sum)

    def line(self,sum):
        return sum
    
    def relu(self,z):
        x = math.log(np.exp(z)+1)
        return np.where(x<10,x,10)

    def deriv_relu(self, z):
        return 1/(1 + np.exp(-1*z))

    def deriv_line(self,z):
        return 1

    def deriv_sigma(self,z):
        return (1 / (1 + np.exp(-1*z)))*(1 - (1 / (1 + np.exp(-1*z))))

    def deriv_hiper(self,z):
        return 1 - (self.hiperbola(z)**2)

    def softmax(self, z):
        e_x = np.exp(z - np.max(z))
        return e_x / e_x.sum(axis=0)

    def softmax_max(self,sums):
        suma=0
        for i in range(len(sums)):
            suma+=np.exp(sums[i])
        return max(sums)/suma

    def choose_fun(self, fun, sum):
        if fun=="sigma":
            return self.sigma(sum)
        elif fun=="relu":
            return self.relu(sum)
        else:
            return self.line(sum)

    def choose_deriv(self, val, fun):
        z = 0
        if fun == "sigma":
            z = self.deriv_sigma(val)
        elif fun == "relu":
            z = self.deriv_relu(val)
        else:
            z = self.deriv_line(val)
        return z