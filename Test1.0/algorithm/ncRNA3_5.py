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
from Crypto.Cipher import ChaCha20  # 导入ChaCha20加密模块

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

# 1. 优化密码子生成，使用生成器而不是预计算数组
def codon_generator():
    for a in 'ACGU':
        for b in 'ACGU':
            for c in 'ACGU':
                yield a + b + c

# 2. 优化替换矩阵生成
def generate_codon_substitution_matrix(seed):
    rng = random.Random(seed)
    codons_list = list(codon_generator())
    shuffled = codons_list.copy()
    rng.shuffle(shuffled)
    # 使用更节省内存的字典推导式
    return {k: v for k, v in zip(codons_list, shuffled)}

# 将 process_chunk 函数移到外部
def process_chunk(chunk_and_matrix):
    """处理密码子块的辅助函数
    
    Args:
        chunk_and_matrix: 元组 (chunk, substitution_matrix)
    Returns:
        list: 替换后的密码子列表
    """
    chunk, substitution_matrix = chunk_and_matrix
    return [substitution_matrix[codon] for codon in chunk]

# 修改后的 substitute_codons 函数
def substitute_codons(codon_sequence, substitution_matrix):
    """使用替换矩阵替换密码子序列
    
    Args:
        codon_sequence: 密码子序列列表
        substitution_matrix: 替换矩阵字典
    Returns:
        list: 替换后的密码子序列
    """
    return [substitution_matrix[codon] for codon in codon_sequence]

# 3. 优化 linear_fold 函数的内存使用
def linear_fold(sequence):
    """使用LinearFold算法进行RNA二级结构预测
    
    参数:
        sequence: RNA序列字符串
    返回:
        结构: 点括号表示的结构字符串
    """
    n = len(sequence)
    # 使用列表替代numpy数组
    dp = [0] * n
    stack = []
    structure = ['.' for _ in range(n)]
    
    for i in range(n):
        while stack and can_pair(sequence[stack[-1]], sequence[i]):
            j = stack.pop()
            if i - j > 3:
                structure[j] = '('
                structure[i] = ')'
                dp[i] = dp[j] + 1
        
        if sequence[i] in 'ACGU':
            stack.append(i)
            
        if len(stack) > 30:
            stack = stack[-30:]
    
    return ''.join(structure)

# 定义 inverse_rna_secondary_structure 函数
def inverse_rna_secondary_structure(codon_sequence, indices_order):
    """逆转 RNA 二级结构重排序
    
    参数:
        codon_sequence: 密码子序列列表
        indices_order: 原始重排序索引
    返回:
        list: 还原后的密码子序列
    """
    # 合并为单个序列
    sequence = ''.join(codon_sequence)
    sequence_array = np.array(list(sequence))
    
    # 转换索引为 numpy 数组
    indices_order_array = np.array(indices_order)
    
    # 验证长度匹配
    if len(indices_order_array) != len(sequence_array):
        # 如果长度不匹配，尝试调整序列长度
        min_len = min(len(indices_order_array), len(sequence_array))
        sequence_array = sequence_array[:min_len]
        indices_order_array = indices_order_array[:min_len]
        print(f"警告：序列长度已调整为 {min_len}")
    
    # 生成逆向索引
    inverse_order = np.argsort(indices_order_array)
    
    # 应用逆向重排序
    original_sequence_array = sequence_array[inverse_order]
    original_sequence = ''.join(original_sequence_array)
    
    # 确保结果长度是3的倍数
    if len(original_sequence) % 3 != 0:
        padding_length = 3 - (len(original_sequence) % 3)
        original_sequence = original_sequence + 'N' * padding_length
        print(f"警告：添加了 {padding_length} 个填充字符以确保序列长度是3的倍数")
    
    # 转换回密码子序列
    original_codon_sequence = [original_sequence[i:i+3] for i in range(0, len(original_sequence), 3)]
    
    return original_codon_sequence

# 4. 优化 apply_rna_secondary_structure 函数
def apply_rna_secondary_structure(codon_sequence):
    """应用 RNA 二级结构重排序
    
    参数:
        codon_sequence: 密码子序列列表
    返回:
        tuple: (重排序后的密码子序列, 索引顺序)
    """
    base_sequence = ''.join(codon_sequence)
    structure = linear_fold(base_sequence)
    
    # 使用列表推导式替代numpy操作
    paired_indices = [i for i, c in enumerate(structure) if c in '()']
    unpaired_indices = [i for i, c in enumerate(structure) if c == '.']
    indices_order = paired_indices + unpaired_indices
    
    # 使用列表操作替代numpy重排序
    new_sequence = ''.join(base_sequence[i] for i in indices_order)
    new_codon_sequence = [new_sequence[i:i+3] for i in range(0, len(new_sequence), 3)]
    
    return new_codon_sequence, indices_order

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

# 流密码加密
def cha_encrypt(data, key):
    """使用 ChaCha20 加密数据
    
    Args:
        data: 要加密的数据（bytes 或 str）
        key: 加密密钥
    Returns:
        bytes: nonce + 加密后的数据
    """
    # 确保输入数据是字节类型
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    else:
        data_bytes = data
        
    cipher = ChaCha20.new(key=key)
    ciphertext = cipher.nonce + cipher.encrypt(data_bytes)
    return ciphertext

# 将 inverse_substitute_codons 函数移到 decrypt 函数之前
def inverse_substitute_codons(codon_sequence, substitution_matrix):
    """逆向替换密���子，将替换后的密码子还原为原始密码子"""
    inverse_matrix = {v: k for k, v in substitution_matrix.items()}
    original_codon_sequence = []
    
    for i, codon in enumerate(codon_sequence):
        if codon not in inverse_matrix:
            raise ValueError(f"无法找到密码子 '{codon}' 的逆替换")
        original_codon_sequence.append(inverse_matrix[codon])
    
    return original_codon_sequence

# 修改 decode_codons_to_plaintext 函数
def decode_codons_to_plaintext(codon_sequence):
    """将密码子序列解码回明文字符串"""
    try:
        # 将密码子序列合并为字符串
        codon_str = ''.join(codon_sequence)
        
        # 确保密码子长度是3的倍数
        if len(codon_str) % 3 != 0:
            padding_length = 3 - (len(codon_str) % 3)
            codon_str = codon_str + 'N' * padding_length
        
        # 拆分为密码子列表
        codons_list = [codon_str[i:i+3] for i in range(0, len(codon_str), 3)]
        
        # 将密码子转换为索引
        codon_indices = []
        for codon in codons_list:
            try:
                idx = np.where(codons == codon)[0][0]
                codon_indices.append(idx)
            except IndexError:
                print(f"警告: 跳过无效密码子 '{codon}'")
                continue
        
        # 转换为Base64字符
        base64_str = ''.join(base64_chars[idx % 64] for idx in codon_indices)
        
        # 添加Base64填充
        padding_length = -len(base64_str) % 4
        base64_padded = base64_str + '=' * padding_length
        
        # 解码Base64
        try:
            plaintext_bytes = base64.b64decode(base64_padded)
            return plaintext_bytes.decode('utf-8')
        except Exception as e:
            print(f"Base64解码失败，尝试其他方法: {str(e)}")
            # 尝试直接解码
            return base64_str
            
    except Exception as e:
        print(f"解码过程出错: {str(e)}")
        raise

# 然后是 decrypt 函数
def decrypt(encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order):
    encrypted_data = verify_and_remove_checksum(encrypted_data_with_checksum)
    dynamic_key = generate_dynamic_key_from_biological_data(seed_sequence, salt)
    decrypted_sequence = cha_decrypt(encrypted_data, dynamic_key)
    unfolded_sequence = inverse_rna_secondary_structure(decrypted_sequence, indices_order)
    original_codon_sequence = inverse_substitute_codons(unfolded_sequence, substitution_matrix)
    plaintext = decode_codons_to_plaintext(original_codon_sequence)
    return plaintext

# 校验和验证
def verify_and_remove_checksum(encrypted_data_with_checksum):
    encrypted_data = encrypted_data_with_checksum[:-32]
    checksum = encrypted_data_with_checksum[-32:]
    computed_checksum = hashlib.sha256(encrypted_data).digest()
    if checksum != computed_checksum:
        raise ValueError("Checksum does not match. Data may be corrupted.")
    return encrypted_data

# AES解密替换为流密码解密
def cha_decrypt(encrypted_data, key):
    """使用 ChaCha20 解密数据
    
    Args:
        encrypted_data: 加密的数据（包含nonce）
        key: 解密密钥
    Returns:
        list: 解密后的密码子序列
    """
    try:
        nonce = encrypted_data[:8]  # ChaCha20 的 nonce 是 8 字节
        ciphertext = encrypted_data[8:]
        cipher = ChaCha20.new(key=key, nonce=nonce)
        decrypted_data = cipher.decrypt(ciphertext)
        
        # 尝试多种编码方式解码
        for encoding in ['utf-8', 'latin1', 'ascii']:
            try:
                data_str = decrypted_data.decode(encoding)
                # 验证解码后的数据是否符合密码子格式
                if len(data_str) % 3 == 0 and all(c in 'ACGU' for c in data_str):
                    codon_sequence = [data_str[i:i+3] for i in range(0, len(data_str), 3)]
                    return codon_sequence
            except UnicodeDecodeError:
                continue
        
        # 如果所有编码都失败，使用二进制方式处理
        data_str = ''.join(chr(b) for b in decrypted_data)
        codon_sequence = [data_str[i:i+3] for i in range(0, len(data_str), 3)]
        return codon_sequence
        
    except Exception as e:
        print(f"解密过程出错: {str(e)}")
        raise

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
        # RSA次只能加密有限长度的数据，需要分块处理
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

# 新增的函数：处理数据块的填充和准备
def prepare_data_chunk(chunk):
    """准备数据块进行加密
    
    Args:
        chunk: 密码子列表
    Returns:
        bytes: 准备好的数据块
    """
    # 将密码子列表连成字符串
    chunk_str = ''.join(chunk)
    # 将字符串编码为字节并填充
    return pad(chunk_str.encode(), AES.block_size)

# 将此函数移到文件前面，放在其他核心函数定义之后，encrypt函数之前
def encode_plaintext_to_codons(plaintext):
    """将明文编码为密码子序列
    
    Args:
        plaintext: 要编码的明文字符串
    Returns:
        list: 密码子序列
    """
    # 将明文转换为base64
    plaintext_bytes = plaintext.encode('utf-8')
    base64_bytes = base64.b64encode(plaintext_bytes)
    base64_str = base64_bytes.decode('ascii').rstrip('=')
    
    # 创建base64字符到索引的映射
    char_to_index = {char: idx for idx, char in enumerate(base64_chars)}
    
    # 将base64字符转换为密码子
    codon_sequence = []
    for char in base64_str:
        try:
            idx = char_to_index[char]
            codon = codons[idx]
            codon_sequence.append(codon)
        except (KeyError, IndexError):
            continue
    
    return codon_sequence

# 修改后的 encrypt 函数
def encrypt(plaintext, seed, seed_sequence, salt):
    try:
        # 1. 编码明文到密码子
        codon_sequence = encode_plaintext_to_codons(plaintext)
        
        # 2. 生成替换矩阵
        substitution_matrix = generate_codon_substitution_matrix(seed)
        
        # 3. 替换密码子
        substituted_sequence = substitute_codons(codon_sequence, substitution_matrix)
        
        # 4. 应用 RNA 二级结构
        structured_sequence, indices_order = apply_rna_secondary_structure(substituted_sequence)
        
        # 5. 生成动态密钥
        dynamic_key = generate_dynamic_key_from_biological_data(seed_sequence, salt)
        
        # 6. 准备加密数据
        data_to_encrypt = ''.join(structured_sequence)
        
        # 7. 使用 ChaCha20 加密
        encrypted_data = cha_encrypt(data_to_encrypt, dynamic_key)
        
        # 8. 添加校验和
        encrypted_data_with_checksum = add_checksum(encrypted_data)
        
        return encrypted_data_with_checksum, substitution_matrix, indices_order
        
    except Exception as e:
        print(f"加密过程出错: {str(e)}")
        raise

# 测试代码
if __name__ == "__main__":
    DEBUG = False  # 设置为 True 时才输出调试信息
    
    # 基本加密测试
    num_threads = multiprocessing.cpu_count()
    plaintext = "I am happy today"
    seed = "123456789"
    seed_sequence = "ACGUACGUACGUACGUACGUACGUACGUACGU"
    salt = b'salt_123'
    
    if DEBUG:
        start_time = time.time()
        encrypted_data_with_checksum, substitution_matrix, indices_order = encrypt(
            plaintext, seed, seed_sequence, salt
        )
        encryption_time = time.time() - start_time
        print(f"加密完成，耗时 {encryption_time:.6f} 秒")
        encrypted_data = encrypted_data_with_checksum[:-32]
        entropy = calculate_entropy(encrypted_data)
        print(f"加密数据的熵: {entropy:.4f} bits/byte")
        plot_entropy_histogram(encrypted_data)

    # 运行性能测试
    print("\n=== 开始性能测试 ===")
    print("1. ncRNA加密/解密性能测试")
    test_performance()
    
    print("\n2. ncRNA、AES和RSA性能对比测试")
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

# 分块处理大文件
def process_large_file(plaintext, chunk_size=1024):
    """分块处理大文件以减少内存使用"""
    chunks = (plaintext[i:i+chunk_size] for i in range(0, len(plaintext), chunk_size))
    results = []
    
    for chunk in chunks:
        encrypted_chunk = encrypt(chunk, seed, seed_sequence, salt)
        results.append(encrypted_chunk)
        
    return combine_results(results)