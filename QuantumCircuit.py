import qiskit
import numpy as np

class QuantumCircuit:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits, backend, shots):
        # ---------------------------
        # Instance variables
        # ---------------------------
        self.n_qubits = n_qubits
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        # Shorthand to access all qubits at once
        all_qubits = [i for i in range(n_qubits)] 
        # Theta is a vector of parameters of length n_qubits 
        self.theta = qiskit.circuit.ParameterVector('theta', n_qubits)
        self.backend = backend
        self.shots = shots
        
        # ---------------------------
        # Circuit definition
        # ---------------------------
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        if n_qubits > 1:
            # Entangle every qubit to the 0th qubit
            self._circuit.cnot(all_qubits[0], all_qubits[1:])
        for i in range(n_qubits):
            self._circuit.ry(self.theta.params[i], all_qubits[i])
        self._circuit.measure_all()
    
    def run(self, thetas):
        # TODO: consider verifying that each list inside thetas has length n_qubits
        binds = []
        for theta in thetas:
            bind = {}
            for i in range(len(theta)):
                bind[self.theta.params[i]] = theta[i]
            binds.append(bind)
        job = qiskit.execute(self._circuit, 
                             self.backend, 
                             shots = self.shots,
                             parameter_binds = binds)
        result = job.result()
        out = []
        for i in range(len(result.results)):
            res = result.get_counts(i)
            print(res)
            counts = np.array(list(res.values()))
            states = np.array(list(res.keys()))
            print(states)
            # Compute probabilities for each state
            probabilities = counts / self.shots
            print(probabilities)
            # Get state expectation
            expectation = []
            for j in range(self.n_qubits):
                exp = 0
                for k in range(len(states)):
                    if states[k][-(j+1)] == '1':
                        exp += probabilities[k]
                expectation.append(exp)
            out.append(expectation)
        return np.array(out)