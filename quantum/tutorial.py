import torch
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import matplotlib.pyplot as plt

# 2. Corrected Quantum Circuit Implementation
class QuantumSolver(tq.QuantumModule):
    def __init__(self, n_wires=4, n_layers=2):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        
        # Trainable parameters
        self.gamma = torch.nn.Parameter(torch.rand(n_layers))
        self.beta = torch.nn.Parameter(torch.rand(n_layers))
        
    def forward(self, qubo_matrix):
        # Convert QUBO matrix to diagonal Hamiltonian
        diag_terms = torch.diag(torch.tensor(qubo_matrix, dtype=torch.float32))
        
        # Initialize quantum device (using state vector simulator)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, record_op=True)
        
        # Initialize superposition state
        for wire in range(self.n_wires):
            tqf.hadamard(self.q_device, wires=wire)
        
        # Build QAOA layer
        for layer in range(self.n_layers):
            # Cost layer: Apply RZ gates to simulate diagonal Hamiltonian
            for wire in range(self.n_wires):
                tqf.rz(self.q_device, wires=wire, params=self.gamma[layer]*diag_terms[wire])
            
            # Mixer layer: Apply RX gates
            for wire in range(self.n_wires):
                tqf.rx(self.q_device, wires=wire, params=self.beta[layer])
        
        # Get final state vector
        state = self.q_device.get_states_1d()
        
        # Corrected energy calculation: Map state probabilities to basis state energies
        prob = torch.abs(state)**2
        basis_energies = torch.tensor([sum([diag_terms[i] * ((idx >> i) & 1) 
                                     for i in range(self.n_wires)])
                                     for idx in range(2**self.n_wires)])
        energy = torch.sum(prob * basis_energies)
        return energy
    
# 3. Evaluation Function
def evaluate_algorithm(qubo_matrix, n_epochs=500):
    model = QuantumSolver(n_wires=qubo_matrix.shape[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    energies = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        energy = model(qubo_matrix)
        (-energy).backward()
        optimizer.step()
        energies.append(energy.item())
    
    plt.figure(figsize=(8,4))
    plt.plot(energies)
    plt.title('Optimization Trajectory')
    plt.xlabel('Epoch')
    plt.ylabel('Energy')
    plt.show()
    
    min_energy = np.min(np.diag(qubo_matrix))
    return np.exp(-(energies[-1] - min_energy))

def generate_ncRNA_qubo(size=6):
    Q = np.array([
        [-124, 88, 0, 44,0,0],
        [88, -124, 88, 0,44,0],
        [0, 88, -124,88,0,44],
        [44, 0, 88, -124,88,0],
        [0, 44, 0, 88,-124,88],
        [0, 0, 44, 0,88,-124]
    ], dtype=np.float32)
    return Q

def generate_RSA_qubo(size=6):
    Q = np.array([
        [-128, 0, 0, 64,64,64],
        [0, -128, 0, 64,64,64],
        [0, 0, -128, 0,0,0],
        [0, 0, 0, -128,0,0],
        [64, 64, 64, 0,-128,0],
        [64, 64, 64, 0,0,-128]
    ], dtype=np.float32)
    return Q

def generate_AES_qubo(size=6):
    Q = np.array([
        [-96, 32, 0, 0,0, 0],
        [32, -96, 0, 0,0, 0],
        [0, 0, -96, 32,0, 0],
        [0, 0, 32, -96,0, 0],
        [0, 0, 0, 0,-96, 32],
        [0, 0, 0, 0,32, -96]
    ], dtype=np.float32)
    return Q

# 4. Main Test Process
if __name__ == "__main__":
    print("Generating QUBO matrices...")
    qubo_ncRNA = generate_ncRNA_qubo()
    qubo_RSA = generate_RSA_qubo()
    qubo_AES = generate_AES_qubo()
    
    print("\nEvaluating Crypto-ncRNA...")
    prob_ncRNA = evaluate_algorithm(qubo_ncRNA)
    
    print("\nEvaluating RSA...")
    prob_RSA = evaluate_algorithm(qubo_RSA)
    
    print("\nEvaluating AES...")
    prob_AES = evaluate_algorithm(qubo_AES)
    
    # Display results
    results = {
        'Crypto-ncRNA': prob_ncRNA,
        'RSA': prob_RSA,
        'AES': prob_AES
    }
    
    plt.figure(figsize=(8,5))
    plt.bar(results.keys(), results.values(), color=['blue', 'orange', 'green'])
    plt.yscale('log')
    plt.ylabel('Success Probability (log scale)')
    plt.title('Quantum Resistance Comparison (6-dim)') # The smaller the better
    plt.show()