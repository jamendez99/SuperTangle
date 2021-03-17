import HybridFunction
from HybridFunction import Hybrid
import torch
import torch.nn as nn

class QuantumNet(nn.Module):
    def __init__(self, n_qubits):
        super(QuantumNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, n_qubits)
        self.hybrid = Hybrid(n_qubits, qiskit.Aer.get_backend('qasm_simulator'), 100, np.pi / 12)
        self.fc3 = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        x = self.fc3(x.float())
        return torch.cat((x, 1 - x), -1)