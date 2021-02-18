#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np
from pennylane.utils import decompose_hamiltonian


def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)

    # QHACK #
    
    w1 = range(len(H.wires))
    w2 = range(len(H.wires))
    dev = qml.device('default.qubit', wires=H.wires)
    dev1 = qml.device('default.qubit', wires=w1)
    dev2 = qml.device('default.qubit', wires=w2)

    def variational_ansatz(params, wires):
        n_qubits = len(wires)
        n_rotations = len(params)

        if n_rotations > 1:
            n_layers = n_rotations // n_qubits
            n_extra_rots = n_rotations - n_layers * n_qubits

            for layer_idx in range(n_layers):
                layer_params = params[layer_idx * n_qubits : layer_idx * n_qubits + n_qubits, :]
                qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
                qml.broadcast(qml.CNOT, wires, pattern="ring")

            extra_params = params[-n_extra_rots:, :]
            extra_wires = wires[: n_qubits - 1 - n_extra_rots : -1]
            qml.broadcast(qml.Rot, extra_wires, pattern="single", parameters=extra_params)
        else:
            qml.Rot(*params[0], wires=wires[0])


    num_qubits = len(H.wires)
    num_param_sets = (2 ** num_qubits) - 1

    energy = 0

    cost_fn = qml.ExpvalCost(variational_ansatz, H, dev)
    opt = qml.AdamOptimizer(stepsize=0.1)
    params = np.random.normal(0, np.pi, (num_param_sets, 3))

    conv_tol = 1e-06
    max_iterations = 200

    for n in range(max_iterations):
        params, prev_energy = opt.step_and_cost(cost_fn, params)
        energy = cost_fn(params)
        conv = np.abs(energy - prev_energy)

        if conv <= conv_tol:
            break

    energies[0] = energy   

    variational_ansatz(params, w1)
    state_1 = dev1.state / energy
    print(state_1)

    # state_1 = np.array([0,1,0])
    
    additional_term = np.outer(state_1, state_1) * 3
    coeffs, ops = decompose_hamiltonian(additional_term)

    new_coeffs = (H.coeffs)
    for c in coeffs:
        new_coeffs.append(c)
    new_ops = (H.ops)
    for o in ops:
        new_ops.append(o)
    H1 = qml.Hamiltonian(new_coeffs, new_ops)

    energy1 = 0

    cost_fn_1 = qml.ExpvalCost(variational_ansatz, H1, dev2)
    params1 = np.random.normal(0, np.pi, (num_param_sets, 3))

    for n in range(max_iterations):
        params1, prev_energy = opt.step_and_cost(cost_fn_1, params1)
        energy1 = cost_fn_1(params1)
        conv = np.abs(energy1 - prev_energy)

        if conv <= conv_tol:
            break

    energies[1] = energy1   

    # QHACK #

    return ",".join([str(E) for E in energies])


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)
