import random
import string
import numpy as np
from qiskit import AerSimulator
from qiskit.aqua.algorithms import Grover
from qiskit.aqua.components.oracles import LogicalExpressionOracle
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd

# 数据长度和加密参数
@dataclass
class EncryptionArgs:
    plaintext: str
    data_length: int
    params: tuple

def generate_random_string(length=32, charset=string.ascii_letters + string.digits):
    """生成随机字符串，用于模拟密钥和密文"""
    return ''.join(random.choices(charset, k=length))

def quantum_attack_simulation(encrypt_function, decrypt_function, plaintext, seed, seed_sequence, salt, max_attempts=1000):
    """
    模拟量子计算暴力破解攻击，攻击密钥和密文
    :param encrypt_function: 加密函数
    :param decrypt_function: 解密函数
    :param plaintext: 明文
    :param seed: 随机数种子
    :param seed_sequence: 随机数序列
    :param salt: 盐值
    :param max_attempts: 最大尝试次数（模拟量子计算的并行性）
    :return: 是否成功破解密文
    """
    # 使用提供的加密函数
    encrypted_data_tuple = encrypt_function(plaintext, seed, seed_sequence, salt)  # 加密
    encrypted_data = encrypted_data_tuple[0]  # 获取加密后的数据
    encryption_params = encrypted_data_tuple[1:]  # 获取加密所需的其他参数（例如密钥）

    # 使用 Qiskit 的 Grover 算法进行量子暴力破解
    oracle_expression = ''.join(['1' if char == '1' else '0' for char in encrypted_data])
    oracle = LogicalExpressionOracle(oracle_expression)

    grover = Grover(oracle)
    backend = AerSimulator()  # 使用最新的 AerSimulator
    result = grover.run(backend)
    
    # 获取量子计算的结果，破解密钥
    measured_key = result['result'][0]

    # 使用解密函数解密并验证是否成功
    decrypted_data = decrypt_function(encrypted_data, seed, seed_sequence, salt, *encryption_params)
    return decrypted_data == plaintext, measured_key

def benchmark_quantum_attack_for_ncRNA(encrypt_function, decrypt_function, data_lengths, key_lengths, runs=10):
    attack_results = {}
    for data_length in data_lengths:
        attack_results[data_length] = {}
        for key_length in key_lengths:
            keyspace = [generate_random_string(key_length) for _ in range(100)]  # 模拟密钥空间
            run_results = []
            for _ in range(runs):
                plaintext = generate_random_string(data_length)
                seed = random.randint(0, 2**32 - 1)
                seed_sequence = generate_random_string(key_length)
                salt = generate_random_string(key_length)
                success, key = quantum_attack_simulation(encrypt_function, decrypt_function, plaintext, seed, seed_sequence, salt)
                run_results.append(success)
            success_rate = np.mean(run_results)
            attack_results[data_length][key_length] = success_rate
    return attack_results

def plot_attack_results(attack_results):
    """绘制量子计算攻击成功率的图表"""
    df = pd.DataFrame.from_dict({(i, j): attack_results[i][j] for i in attack_results for j in attack_results[i]}, 
                                orient='index', columns=['Success Rate'])
    df.index = pd.MultiIndex.from_tuples(df.index, names=['Data Length', 'Key Length'])
    df = df.reset_index()

    # 绘制图表
    plt.figure(figsize=(12, 8))
    for key_length in df['Key Length'].unique():
        subset = df[df['Key Length'] == key_length]
        plt.plot(subset['Data Length'], subset['Success Rate'], label=f'Key Length {key_length}')
    
    plt.xlabel('Data Length (Bytes)')
    plt.ylabel('Attack Success Rate')
    plt.title('Quantum Computing Attack Simulation on ncRNA Algorithm')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    data_lengths = [70, 100, 500, 1000, 5000, 10000]
    key_lengths = [16, 32, 64]  # 假设的不同密钥长度
    runs = 10

    # 从 ncRNA 加密算法中引入加密和解密函数
    from algorithm.ncRNA3_5 import encrypt as ncRNA_encrypt, decrypt as ncRNA_decrypt

    # 进行抗量子计算攻击测试
    print(f"Testing ncRNA algorithm for quantum computing attack simulation")
    attack_results = benchmark_quantum_attack_for_ncRNA(ncRNA_encrypt, ncRNA_decrypt, data_lengths, key_lengths, runs)
    plot_attack_results(attack_results)

if __name__ == "__main__":
    main()
