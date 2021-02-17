#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pennylane as qml
from pennylane import numpy as np
import sys

def simple_circuits_20(angle):
    """The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.
    In this function:
        * Rotate the qubit around the x-axis by angle
        * Measure the probability the qubit is in the zero state
    Args:
        angle (float): how much to rotate a state around the x-axis
    Returns:
        float: the probability of of the state being in the 0 ground state
    """
    prob = 0.0
    # QHACK #

    # Step 1 : initalize a device
    dev1=qml.device('default.qubit',wires=['wire1'])

    # Step 2 : Create a quantum circuit and qnode
    def rotation(angle):
        qml.RX(angle,wires='wire1')
        return qml.probs(wires='wire1')
        

    # Step 3 : Run the qnode
    circuit=qml.QNode(rotation,dev1)
    # prob = ?
    prob=circuit(angle)[0]

    # QHACK #
    return prob
if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process input
    angle_str = sys.stdin.read()
    angle = float(angle_str)

    ans = simple_circuits_20(angle)

    if isinstance(ans, np.tensor):
        ans = ans.item()

    if not isinstance(ans, float):
        raise TypeError("the simple_circuits_20 function needs to return a float")

    print(ans)


# In[95]:





# In[96]:





# In[ ]:




