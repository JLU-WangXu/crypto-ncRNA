import numpy as np
import csv

# ======================== Common utility functions ========================
def save_qubo(matrix, filename):
    """Save the QUBO matrix as a CSV file (force 8-bit integer)"""
    clipped = np.clip(matrix, -128, 127).astype(int)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(clipped)
    print(f"File generated: {filename}")

# ======================== Algorithm QUBO generator ========================
def generate_ncRNA_LWE(n=5, bits=5):
    """Generate the LWE problem QUBO matrix for ncRNA (banded symmetric structure)"""
    size = n * bits
    Q = np.zeros((size, size))

    # Core parameters (simulating RNA folding characteristics)
    diag_base = -128
    band_value = 88
    decay = 0.5

    for i in range(size):
        # Diagonal: Simulate energy gradients in different dimensions
        Q[i,i] = diag_base + 4 * (i % bits)

        # Band coupling: Simulate RNA structural constraints
        for offset in [1, bits]:
            if i + offset < size:
                val = int(band_value * (decay ** (offset//bits)))
                Q[i, i+offset] = val
                Q[i+offset, i] = val

    return Q

def generate_RSA_factorization(prime_bits=50):
    """Generate the QUBO matrix for the RSA factorization problem (simplified to 100 bits)"""
    size = 100  # 50-bit p + 50-bit q
    Q = np.zeros((size, size))

    # Diagonal: Bit constraints for large number factorization
    np.fill_diagonal(Q, -128)

    # Off-diagonal: Product constraints (high-density coupling)
    for i in range(50):
        for j in range(50):
            Q[i, 50+j] = 64  # Simulate p_i * q_j terms
            Q[50+j, i] = 64

    return np.clip(Q, -128, 127)

def generate_AES_keysearch(key_bits=256):
    """Generate the QUBO matrix for AES key search (diagonal model)"""
    Q = np.diag([-128]*key_bits)  # Diagonal terms represent key bit constraints

    # Add weak coupling terms (simulating ciphertext matching)
    for i in range(0, key_bits, 8):
        Q[i:i+8, i:i+8] += 32  # Weak coupling per byte

    return np.clip(Q, -128, 127)

# ======================== Generation and Testing ========================
if __name__ == "__main__":
    # Generate QUBO matrices
    qubo_ncRNA = generate_ncRNA_LWE(n=5, bits=5)   # 25x25
    qubo_RSA = generate_RSA_factorization()        # 100x100
    qubo_AES = generate_AES_keysearch()            # 256x256

    # Save files
    save_qubo(qubo_ncRNA, "ncRNA_LWE_25x25.csv")
    save_qubo(qubo_RSA, "RSA_100bit.csv")
    save_qubo(qubo_AES, "AES_256bit.csv")