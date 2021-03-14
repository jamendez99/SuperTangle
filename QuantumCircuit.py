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
    
    def run(self, thetas_list):
        bind_dict = {}
        binds = []
#         for i in range(len(self.theta.params)):
#             for theta in thetas:
#                 bind_dict[self.theta.params] = theta
#         bounds_circuits = [self._circuit.bind_parameters({self.theta: theta} for theta in thetas)]
        for i in range(len(self.theta.params)):
#             for thetas in thetas_list:
            binds.append({self.theta.params[i] : thetas[i] for thetas in thetas_list})
#                 bind_dict[self.theta.params[i]] = thetas[i]
        job = qiskit.execute(self._circuit, 
                             self.backend, 
                             shots = self.shots,
#                              parameter_binds = [bind_dict])
                             parameter_binds = binds)
#                              parameter_binds = [{self.theta: theta} for theta in thetas])
        result = job.result().get_counts(self._circuit)
        
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        
        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)
        
        return np.array([expectation])