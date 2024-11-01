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
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Random import get_random_bytes  # 添加这行导入语句在文件顶部的导入区域

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

# 4. 替换原有的nussinov_algorithm函数
def linear_fold(sequence):
    """使用LinearFold算法进行RNA二级结构预测
    
    参数:
        sequence: RNA序列字符串
    返回:
        结构: 点括号表示的结构字符串
    """
    n = len(sequence)
    # 使用O(n)空间的动态规划数组
    dp = np.zeros(n, dtype=np.int8)
    # 使用栈来存储可能的碱基对
    stack = []
    structure = np.full(n, '.', dtype='U1')
    
    # 线性扫描序列
    for i in range(n):
        while stack and can_pair(sequence[stack[-1]], sequence[i]):
            j = stack.pop()
            if i - j > 3:  # 确保最小环尺寸
                structure[j] = '('
                structure[i] = ')'
                dp[i] = dp[j] + 1
        
        # 只将可能配对的碱基位置加入栈
        if sequence[i] in 'ACGU':
            stack.append(i)
            
        # 定期清理栈中不可能再配对的位置
        if len(stack) > 30:  # 设置合理的窗口大小
            stack = stack[-30:]
    
    return ''.join(structure)

# 更新apply_rna_secondary_structure函数以使用新的linear_fold
def apply_rna_secondary_structure(codon_sequence):
    base_sequence = ''.join(codon_sequence)
    # 使用LinearFold算法
    structure = linear_fold(base_sequence)
    
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
    # 创建个字典来映射codon到Base64索引
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

def test_comparison():
    """比较ncRNA、AES和RSA的性能"""
    plaintext_lengths = [50, 100, 200, 400, 800, 1600]
    ncrna_times = []
    aes_times = []
    rsa_times = []
    
    # 初始化密钥
    seed = "123456789"
    seed_sequence = "ACGUACGUACGUACGUACGUACGUACGUACGU"
    salt = b'salt_123'
    aes_key = get_random_bytes(32)
    rsa_key = RSA.generate(2048)
    rsa_cipher = PKCS1_OAEP.new(rsa_key)
    
    def test_ncrna(plaintext):
        start_time = time.time()
        encrypted_data, _, _ = encrypt(plaintext, seed, seed_sequence, salt)
        return time.time() - start_time
    
    def test_aes(plaintext):
        start_time = time.time()
        cipher = AES.new(aes_key, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(pad(plaintext.encode(), AES.block_size))
        return time.time() - start_time
    
    def test_rsa(plaintext):
        start_time = time.time()
        # RSA每次只能加密有限长度的数据，需要分块处理
        block_size = 190  # RSA-2048的最大加密块大小
        blocks = [plaintext[i:i+block_size].encode() for i in range(0, len(plaintext), block_size)]
        for block in blocks:
            rsa_cipher.encrypt(block)
        return time.time() - start_time

    for length in plaintext_lengths:
        test_text = 'A' * length
        ncrna_times.append(test_ncrna(test_text))
        aes_times.append(test_aes(test_text))
        rsa_times.append(test_rsa(test_text))

    # 绘制性能对比图
    plt.figure(figsize=(10, 6))
    plt.plot(plaintext_lengths, ncrna_times, 'o-', label='ncRNA')
    plt.plot(plaintext_lengths, aes_times, 's-', label='AES')
    plt.plot(plaintext_lengths, rsa_times, '^-', label='RSA')
    plt.xlabel('Plaintext Length (characters)')
    plt.ylabel('Encryption Time (seconds)')
    plt.title('Encryption Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 打印详细结果
    print("\n性能对比结果:")
    print("明文长度\tncRNA(s)\tAES(s)\t\tRSA(s)")
    print("-" * 50)
    for i, length in enumerate(plaintext_lengths):
        print(f"{length}\t\t{ncrna_times[i]:.6f}\t{aes_times[i]:.6f}\t{rsa_times[i]:.6f}")

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
    
    print("\n开始性能对比测试...")
    test_comparison()

def chunked_encryption(plaintext, chunk_size=400):
    chunks = [plaintext[i:i+chunk_size] for i in range(0, len(plaintext), chunk_size)]
    encrypted_chunks = []
    for chunk in chunks:
        # 处理每个较小的块
        encrypted_chunk = encrypt_chunk(chunk)
        encrypted_chunks.append(encrypted_chunk)
    return combine_chunks(encrypted_chunks)

def optimized_nussinov(sequence):
    # 使用稀疏动态规划
    # 只存储可能配对的位置
    pairs = {}
    for i in range(len(sequence)):
        for j in range(i + 4, len(sequence)):  # 最小环尺寸为4
            if can_pair(sequence[i], sequence[j]):
                pairs[(i,j)] = True
    # 只在可能配对的位置进行计算