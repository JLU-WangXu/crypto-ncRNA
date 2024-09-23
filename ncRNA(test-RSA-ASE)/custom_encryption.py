import random
import hashlib
import time
import matplotlib.pyplot as plt

# 1. 信息转录与编码：通过置换矩阵模拟DNA到RNA的转录过程
def generate_substitution_matrix(seed):
    random.seed(seed)
    bases = ['A', 'C', 'G', 'T']  # 碱基
    substitution = dict()
    shuffled_bases = random.sample(bases, len(bases))
    for i, base in enumerate(bases):
        substitution[base] = shuffled_bases[i]
    return substitution

def transcribe_dna_to_rna(plaintext, substitution_matrix):
    transcribed = ''.join([substitution_matrix.get(char, char) for char in plaintext])
    return transcribed

# 2. 剪接与数据重组：模拟mRNA剪接，随机选择不同数据块重新组合
def split_and_splice(data, block_size=3):
    blocks = [data[i:i+block_size] for i in range(0, len(data), block_size)]
    original_order = blocks.copy()  # 保存原始顺序以用于解密
    random.shuffle(blocks)  # 随机排列数据块，模拟可变剪接
    spliced_data = ''.join(blocks)
    return spliced_data, original_order

def inverse_splice(data, original_order):
    return ''.join(original_order)

# 3. miRNA调控与动态密钥生成
def generate_dynamic_key(seed=None):
    if seed is None:
        seed = int(time.time())  # 使用时间戳作为种子
    seed_str = str(seed)
    hash_object = hashlib.sha256(seed_str.encode())
    dynamic_key = int(hash_object.hexdigest(), 16) % (2**128)
    return dynamic_key

def apply_dynamic_key(data, key):
    key_bin = format(key, '0128b')
    data_bin = ''.join(format(ord(char), '08b') for char in data)
    encrypted_data = ''.join('1' if data_bin[i] != key_bin[i % len(key_bin)] else '0' for i in range(len(data_bin)))
    chars = [chr(int(encrypted_data[i:i+8], 2)) for i in range(0, len(encrypted_data), 8)]
    return ''.join(chars)

def reverse_dynamic_key(data, key):
    return apply_dynamic_key(data, key)

# 4. 数据编辑与冗余保护
def insert_redundancy(encrypted_data):
    redundancy = ''.join(random.choice('01') for _ in range(8))
    return encrypted_data + redundancy

# 自定义加密过程
def custom_encrypt(data, seed=None):
    random.seed(seed)
    encrypted = list(data)
    random.shuffle(encrypted)
    return ''.join(encrypted)

# 自定义解密过程
def custom_decrypt(encrypted_data, seed, original_order):
    start_time = time.time()
    encrypted_data = encrypted_data[:-8]
    dynamic_key = generate_dynamic_key(seed)
    decrypted_spliced_data = reverse_dynamic_key(encrypted_data, dynamic_key)
    decrypted_data = inverse_splice(decrypted_spliced_data, original_order)
    decryption_time = time.time() - start_time
    return decrypted_data, decryption_time

# 测试自定义算法
def test_custom_encryption(num_tests=10):
    data = "ACGTACGTACGT" * 10
    seed = "123456789"
    encryption_times = []
    decryption_times = []

    for _ in range(num_tests):
        encrypted_data, original_order, encryption_time = custom_encrypt(data, seed)
        decrypted_data, decryption_time = custom_decrypt(encrypted_data, seed, original_order)

        encryption_times.append(encryption_time)
        decryption_times.append(decryption_time)

    avg_encryption_time = sum(encryption_times) / len(encryption_times)
    avg_decryption_time = sum(decryption_times) / len(decryption_times)

    print(f"Custom Algorithm - 平均加密时间：{avg_encryption_time:.6f} 秒")
    print(f"Custom Algorithm - 平均解密时间：{avg_decryption_time:.6f} 秒")

    plt.figure(figsize=(10, 5))
    plt.plot(encryption_times, label='Encryption Time')
    plt.plot(decryption_times, label='Decryption Time')
    plt.xlabel('Test Number')
    plt.ylabel('Time (seconds)')
    plt.title('Custom Algorithm Encryption and Decryption Time per Test')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_custom_encryption()
