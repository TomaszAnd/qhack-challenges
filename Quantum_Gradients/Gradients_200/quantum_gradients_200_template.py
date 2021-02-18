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
    s = 0.5 * np.pi
    denom = 4 * np.sin(s) ** 2
    shift = np.eye(len(weights))

    # QHACK #
    def parameter_shift_term(qnode, weights, i):
        shifted = weights.copy()
        shifted[i] += np.pi/4
        forward = qnode(shifted)  # forward evaluation

        shifted[i] -= np.pi/2
        backward = qnode(shifted) # backward evaluation
        return 0.5 * ((forward - backward)/(np.sin(np.pi/4)))
    
    def parameter_shift(qnode, params):
        for i in range(len(params)):
            gradient[i] = parameter_shift_term(qnode, weights, i)
        return gradient
    def hess_gen_results(weights):
        
        results = {}
        import itertools
        for c in itertools.combinations(range(len(weights)), r=2):
            if not results.get(c):
                weights_pp = weights + s * (shift[c[0]] + shift[c[1]])
                weights_pm = weights + s * (shift[c[0]] - shift[c[1]])
                weights_mp = weights - s * (shift[c[0]] - shift[c[1]])
                weights_mm = weights - s * (shift[c[0]] + shift[c[1]])

                f_pp = circuit(weights_pp)
                f_pm = circuit(weights_pm)
                f_mp = circuit(weights_mp)
                f_mm = circuit(weights_mm)
                results[c] = (f_pp, f_mp, f_pm, f_mm)
        for i in range(len(weights)):
            if not results.get((i, i)):
                f_p = circuit(weights + 0.5 * np.pi * shift[i])
                f_m = circuit(weights - 0.5 * np.pi * shift[i])
                f = circuit(weights)

                results[(i, i)] = (f_p, f_m, f)
        return results


    def get_hess(weights):
        results=hess_gen_results(weights)
        import itertools
        hessian = np.zeros([5, 5], dtype=np.float64)
        for c in itertools.combinations(range(len(weights)), r=2):
            r = results[c]
            hessian[c] = (r[0] - r[1] - r[2] + r[3]) / denom

        hessian = hessian + hessian.T

        for i in range(len(weights)):
            r = results[(i, i)]
            hessian[i, i] = (r[0] + r[1] - 2 * r[2]) / 2
        return hessian

    gradient=parameter_shift(circuit,weights)
    hessian=get_hess(weights)
    
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        

                
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




