import hashlib  # 导入hashlib模块用于生成哈希值
import random   # 导入random模块用于生成随机数
import time     # 导入time模块用于计算时间
from collections import Counter  # 导入Counter用于计数对象
from Crypto.Cipher import AES  # 导入AES加密模块
from Crypto.Util.Padding import pad, unpad  # 导入填充和去填充函数
import base64   # 导入base64模块用于编码
import math     # 导入math模块用于数学运算
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot用于绘图
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256  # 导入正确的哈希模块
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# 定义64个密码子
codons = np.array([a + b + c for a in 'ACGU' for b in 'ACGU' for c in 'ACGU'])

# 定义Base64字符集（不包括'='）
base64_chars = np.array(list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'))

# 添加 can_pair 函数在 nussinov_algorithm 函数之前
def can_pair(base1, base2):
    """检查两个RNA碱基是否可以配对
    
    允许的碱基对: AU, UA, GC, CG, GU, UG
    """
    pairs = {
        ('A', 'U'), ('U', 'A'),
        ('G', 'C'), ('C', 'G'),
        ('G', 'U'), ('U', 'G')
    }
    return (base1, base2) in pairs

# 1. 优化 generate_codon_substitution_matrix
def generate_codon_substitution_matrix(seed):
    # 使用预计算的codons数组和更高效的随机打乱
    rng = np.random.default_rng(int(seed))
    # 直接在内存中打乱，避免创建新数组
    shuffled_codons = codons.copy()
    rng.shuffle(shuffled_codons)
    return dict(zip(codons, shuffled_codons))

# 2. 优化 encode_plaintext_to_codons
def encode_plaintext_to_codons(plaintext, num_threads=None):
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()
    
    # 预计算 base64 字符到索引的映射
    char_to_index = np.zeros(128, dtype=np.int8)
    for idx, char in enumerate(base64_chars):
        char_to_index[ord(char)] = idx
    
    # 使用 numpy 进行向量化操作
    plaintext_bytes = plaintext.encode('utf-8')
    base64_bytes = base64.b64encode(plaintext_bytes)
    base64_str = base64_bytes.decode('ascii').rstrip('=')
    
    # 直接将字符串转换为索引数组
    indices = char_to_index[[ord(c) for c in base64_str]]
    
    # 使用向量化操作获取密码子
    return codons[indices].tolist()

# 3. 密码子替换
def substitute_codons(codon_sequence, substitution_matrix, num_threads=None):
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()
    
    chunk_size = max(1, len(codon_sequence) // num_threads)
    chunks = [codon_sequence[i:i + chunk_size] for i in range(0, len(codon_sequence), chunk_size)]
    
    def process_chunk(chunk):
        return [substitution_matrix[codon] for codon in chunk]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_chunk, chunks))
    
    return [codon for chunk in results for codon in chunk]

# 4. 优化 nussinov_algorithm
@lru_cache(maxsize=2048)  # 增加缓存大小
def nussinov_algorithm(sequence):
    n = len(sequence)
    # 使用更小的数据类型
    dp = np.zeros((n, n), dtype=np.int8)
    structure = np.full(n, '.', dtype='U1')
    
    # 使用预计算的配对表优化 can_pair 检查
    pair_lookup = np.zeros((128, 128), dtype=bool)
    for b1, b2 in [('A','U'), ('U','A'), ('G','C'), ('C','G'), ('G','U'), ('U','G')]:
        pair_lookup[ord(b1), ord(b2)] = True
    
    # 使用numba或向量化操作优化循环
    for k in range(1, n):
        for i in range(n - k):
            j = i + k
            if can_pair(sequence[i], sequence[j]):
                dp[i,j] = max(dp[i+1,j], dp[i,j-1], dp[i+1,j-1] + 1)
            else:
                dp[i,j] = max(dp[i+1,j], dp[i,j-1])
    
    # 优化回溯过程
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if i >= j: continue
        if dp[i,j] == dp[i+1,j]:
            stack.append((i+1, j))
        elif dp[i,j] == dp[i,j-1]:
            stack.append((i, j-1))
        elif can_pair(sequence[i], sequence[j]) and dp[i,j] == dp[i+1,j-1] + 1:
            structure[i] = '('
            structure[j] = ')'
            stack.append((i+1, j-1))
    
    return ''.join(structure)

def apply_rna_secondary_structure(codon_sequence):
    base_sequence = ''.join(codon_sequence)
    structure = nussinov_algorithm(base_sequence)
    
    # 使用numpy的布尔索引
    structure_array = np.array(list(structure))
    paired_mask = (structure_array == '(') | (structure_array == ')')
    indices_order = np.concatenate([
        np.where(paired_mask)[0],
        np.where(~paired_mask)[0]
    ])
    
    sequence_array = np.array(list(base_sequence))
    new_sequence = ''.join(sequence_array[indices_order])
    new_codon_sequence = [new_sequence[i:i+3] for i in range(0, len(new_sequence), 3)]
    return new_codon_sequence, indices_order.tolist()

# 5. 从生物数据生成动态密钥
def generate_dynamic_key_from_biological_data(seed_sequence, salt, iterations=100000):
    valid_bases = set('ACGU')
    if not set(seed_sequence.upper()).issubset(valid_bases):
        raise ValueError("Seed sequence contains invalid bases.")
    dynamic_key = PBKDF2(seed_sequence, salt, dkLen=32, count=iterations, hmac_hash_module=SHA256)
    return dynamic_key

# 6. AES加密
def aes_encrypt(data_sequence, key):
    data_str = ''.join(data_sequence)
    data_bytes = data_str.encode()
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(pad(data_bytes, AES.block_size))
    return cipher.nonce + tag + ciphertext

# 7. 添加校验和
def add_checksum(encrypted_data):
    checksum = hashlib.sha256(encrypted_data).digest()
    return encrypted_data + checksum

# 加密函数
def encrypt(plaintext, seed, seed_sequence, salt, num_threads=None):
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()
        
    substitution_matrix = generate_codon_substitution_matrix(seed)
    codon_sequence = encode_plaintext_to_codons(plaintext, num_threads)
    substituted_sequence = substitute_codons(codon_sequence, substitution_matrix, num_threads)
    folded_sequence, indices_order = apply_rna_secondary_structure(substituted_sequence)
    dynamic_key = generate_dynamic_key_from_biological_data(seed_sequence, salt)
    encrypted_data = aes_encrypt(folded_sequence, dynamic_key)
    encrypted_data_with_checksum = add_checksum(encrypted_data)
    return encrypted_data_with_checksum, substitution_matrix, indices_order

# 校验和验证
def verify_and_remove_checksum(encrypted_data_with_checksum):
    encrypted_data = encrypted_data_with_checksum[:-32]
    checksum = encrypted_data_with_checksum[-32:]
    computed_checksum = hashlib.sha256(encrypted_data).digest()
    if checksum != computed_checksum:
        raise ValueError("Checksum does not match. Data may be corrupted.")
    return encrypted_data

# AES解密
def aes_decrypt(encrypted_data, key):
    nonce = encrypted_data[:16]
    tag = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    decrypted_padded = cipher.decrypt_and_verify(ciphertext, tag)
    decrypted_data = unpad(decrypted_padded, AES.block_size)
    data_str = decrypted_data.decode()
    codon_sequence = [data_str[i:i+3] for i in range(0, len(data_str), 3)]
    return codon_sequence

# 修正 inverse_rna_secondary_structure 函数，确保返回的 original_codon_sequence 是字符串列表
def inverse_rna_secondary_structure(codon_sequence, indices_order):
    base_sequence = ''.join(codon_sequence)
    original_sequence = np.array(['A'] * len(base_sequence))  # 使用'A'作为默认填充值
    original_sequence[indices_order[:len(base_sequence)]] = list(base_sequence)
    original_codon_sequence = [''.join(original_sequence[i:i+3]) for i in range(0, len(original_sequence), 3) if ''.join(original_sequence[i:i+3])]
    return original_codon_sequence

# 修正 inverse_substitute_codons 函数，确保 substitution_matrix 的值是字符串
def inverse_substitute_codons(codon_sequence, substitution_matrix):
    inverse_substitution_matrix = {v: k for k, v in substitution_matrix.items()}
    substituted_array = np.array(codon_sequence)
    vectorized_inverse = np.vectorize(lambda x: inverse_substitution_matrix.get(x, 'A'))  # 使用'A'作为默认
    original_codon_sequence = vectorized_inverse(substituted_array).tolist()
    return original_codon_sequence

# 将密码子序列解码为明文
def decode_codons_to_plaintext(codon_sequence):
    # 创建一个字典来映射codon到Base64索引
    codon_to_index = {codon: idx for idx, codon in enumerate(codons)}
    try:
        indices = [codon_to_index[codon] for codon in codon_sequence]
    except KeyError:
        # 如果存在未定义的codon，返回空字符串或处理错误
        return ""
    base64_str = ''.join(base64_chars[indices])
    # 补齐Base64的填充字符
    base64_str += '=' * (-len(base64_str) % 4)
    try:
        plaintext_bytes = base64.b64decode(base64_str)
        plaintext = plaintext_bytes.decode('utf-8')
    except Exception:
        plaintext = ""
    return plaintext

# 解密函数
def decrypt(encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order):
    encrypted_data = verify_and_remove_checksum(encrypted_data_with_checksum)
    dynamic_key = generate_dynamic_key_from_biological_data(seed_sequence, salt)
    decrypted_sequence = aes_decrypt(encrypted_data, dynamic_key)
    unfolded_sequence = inverse_rna_secondary_structure(decrypted_sequence, indices_order)
    original_codon_sequence = inverse_substitute_codons(unfolded_sequence, substitution_matrix)
    plaintext = decode_codons_to_plaintext(original_codon_sequence)
    return plaintext

# 计算熵
def calculate_entropy(data):
    byte_data = data
    if isinstance(data, str):
        byte_data = data.encode()
    data_length = len(byte_data)
    if data_length == 0:
        return 0.0
    frequencies = Counter(byte_data)
    freqs = np.array(list(frequencies.values()), dtype=float)
    probabilities = freqs / data_length
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# 绘制熵直方图
def plot_entropy_histogram(data):
    byte_data = data
    if isinstance(data, str):
        byte_data = data.encode()
    frequencies = Counter(byte_data)
    bytes_list = np.array(list(frequencies.keys()))
    counts = np.array(list(frequencies.values()))
    plt.bar(bytes_list, counts)
    plt.xlabel('Byte Value')
    plt.ylabel('Frequency')
    plt.title('Byte Frequency Distribution of Encrypted Data')
    plt.show()

# 性能测试函数
def test_performance():
    plaintext_lengths = [50, 100, 200, 400, 800, 1600]
    encryption_times = []
    decryption_times = []
    seed = "123456789"
    seed_sequence = "ACGUACGUACGUACGUACGUACGUACGUACGU"
    salt = b'salt_123'

    def process_length(length):
        plaintext = 'A' * length
        start_time = time.time()
        encrypted_data_with_checksum, substitution_matrix, indices_order = encrypt(plaintext, seed, seed_sequence, salt)
        encryption_time = time.time() - start_time
        start_time = time.time()
        decrypted_plaintext = decrypt(encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order)
        decryption_time = time.time() - start_time
        return (encryption_time, decryption_time)

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_length, plaintext_lengths)
        for et, dt in results:
            encryption_times.append(et)
            decryption_times.append(dt)

    plt.plot(plaintext_lengths, encryption_times, label='Encryption Time')
    plt.plot(plaintext_lengths, decryption_times, label='Decryption Time')
    plt.xlabel('Plaintext Length (characters)')
    plt.ylabel('Time (seconds)')
    plt.title('Encryption and Decryption Time vs. Plaintext Length')
    plt.legend()
    plt.show()

# 主测试代码
if __name__ == "__main__":
    # 使用系统CPU核心数
    num_threads = multiprocessing.cpu_count()
    print(f"使用 {num_threads} 个线程进行并行处理")
    
    plaintext = "Hello, World! This is a test of the encryption algorithm based on ncRNA."
    seed = "123456789"
    seed_sequence = "ACGUACGUACGUACGUACGUACGUACGUACGU"
    salt = b'salt_123'
    
    # 添加开始时间记录
    start_time = time.time()
    
    # 使用并行处理进行加密
    encrypted_data_with_checksum, substitution_matrix, indices_order = encrypt(
        plaintext, seed, seed_sequence, salt, num_threads
    )
    
    encryption_time = time.time() - start_time
    print(f"Encryption completed in {encryption_time:.6f} seconds")
    encrypted_data = encrypted_data_with_checksum[:-32]
    entropy = calculate_entropy(encrypted_data)
    print(f"Entropy of encrypted data: {entropy:.4f} bits/byte")
    plot_entropy_histogram(encrypted_data)
    start_time = time.time()
    decrypted_plaintext = decrypt(encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order)
    decryption_time = time.time() - start_time
    print(f"Decryption completed in {decryption_time:.6f} seconds")
    print("Decrypted Plaintext:", decrypted_plaintext)
    if plaintext == decrypted_plaintext:
        accuracy = 100.0
        print("Encryption and decryption successful. Accuracy: 100%")
    else:
        accuracy = 0.0
        print("Encryption and decryption failed. Accuracy: 0%")
    print("\nTest Summary:")
    print(f"Plaintext Length: {len(plaintext)} characters")
    print(f"Encrypted Data Length: {len(encrypted_data_with_checksum)} bytes")
    print(f"Encryption Time: {encryption_time:.6f} seconds")
    print(f"Decryption Time: {decryption_time:.6f} seconds")
    print(f"Entropy of Encrypted Data: {entropy:.4f} bits/byte")
    print(f"Accuracy: {accuracy}%")