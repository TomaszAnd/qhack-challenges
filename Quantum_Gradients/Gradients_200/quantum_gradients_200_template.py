#!/usr/bin/env python
# coding: utf-8

# In[35]:


import sys
import pennylane as qml
import numpy as np

def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.
    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.
    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).
            * gradient is a real NumPy array of size (5,).
            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)
    
    

    # QHACK #
    s = np.pi/2
    forward_values = np.zeros([5], dtype=np.float64)
    backward_values = np.zeros([5], dtype=np.float64)
    noshift = circuit(weights)
    
    def parameter_shift_term(qnode, weights, i):
        shifted = weights.copy()
        shifted[i] += s
        forward = qnode(shifted)  # forward evaluation
        forward_values[i]=forward

        shifted[i] -= 2*s
        backward = qnode(shifted) # backward evaluation
        backward_values[i]=backward
        return 0.5 * ((forward - backward)/(np.sin(s)))
    
    def get_gradient(qnode, params):
        for i in range(len(params)):
            gradient[i] = parameter_shift_term(qnode, params, i)

    def get_hessian(qnode, weights):
        ij = []
        shifted = weights.copy()
        for i in range(len(weights)-1):
            for j in range(1, len(weights)):
                if i!=j and [j,i] not in ij:
                    shifted[i] += s
                    shifted[j] += s
                    pp = qnode(shifted)
                    shifted[j] -= 2*s
                    pm = qnode(shifted)
                    shifted[i] -= 2*s
                    mm = qnode(shifted)
                    shifted[j] += 2*s
                    mp = qnode(shifted)
                    hessian[i][j] = (pp + mm - pm - mp)/(4*(np.sin(s)**2))
                    hessian[j][i] = hessian[i][j]
                    shifted[i] += s
                    shifted[j] -= s
                ij.append([i,j])
        for i in range(len(weights)):
            hessian[i][i] = (forward_values[i] + backward_values[i] - 2*noshift)/2
    
    get_gradient(circuit,weights)
    get_hessian(circuit, weights)

    hessian = hessian + 1 - 1

    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]

if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




