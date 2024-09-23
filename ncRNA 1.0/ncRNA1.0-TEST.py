import random
import hashlib
import datetime
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
        now = datetime.datetime.now()
        seed = int(now.strftime('%Y%m%d%H%M%S'))

    seed_str = str(seed)
    hash_object = hashlib.sha256(seed_str.encode())
    dynamic_key = int(hash_object.hexdigest(), 16) % (2**128)
    
    return dynamic_key

def apply_dynamic_key(data, key):
    key_bin = format(key, '0128b')
    data_bin = ''.join(format(ord(char), '08b') for char in data)  # 将数据转换为二进制格式
    
    # 使用异或操作将数据与密钥进行加密
    encrypted_data = ''.join('1' if data_bin[i] != key_bin[i % len(key_bin)] else '0' for i in range(len(data_bin)))
    
    # 将二进制数据转换回字符
    chars = [chr(int(encrypted_data[i:i+8], 2)) for i in range(0, len(encrypted_data), 8)]
    return ''.join(chars)

def reverse_dynamic_key(data, key):
    # 和apply_dynamic_key相同，异或操作是对称的，解密时调用相同逻辑
    return apply_dynamic_key(data, key)

# 4. 数据编辑与冗余保护
def insert_redundancy(encrypted_data):
    redundancy = ''.join(random.choice('01') for _ in range(8))  # 8位随机冗余
    return encrypted_data + redundancy

# 加密过程
def encrypt(plaintext, seed=None):
    start_time = time.time()  # 开始计时
    if seed is None:
        seed = "initial_seed_value"
    substitution_matrix = generate_substitution_matrix(seed)
    
    transcribed_data = transcribe_dna_to_rna(plaintext, substitution_matrix)
    print(f"转录后的数据：{transcribed_data}")
    
    spliced_data, original_order = split_and_splice(transcribed_data)
    print(f"剪接后的数据：{spliced_data}")
    
    dynamic_key = generate_dynamic_key(seed=int(seed))
    print(f"动态密钥：{dynamic_key}")
    
    encrypted_data = apply_dynamic_key(spliced_data, dynamic_key)
    print(f"应用动态密钥后的加密数据：{encrypted_data}")
    
    encrypted_with_redundancy = insert_redundancy(encrypted_data)
    print(f"加密后的数据（含冗余）：{encrypted_with_redundancy}")
    
    end_time = time.time()  # 结束计时
    encryption_time = end_time - start_time  # 计算加密时间
    print(f"加密时间：{encryption_time:.6f} 秒")
    
    return encrypted_with_redundancy, original_order, encryption_time

# 解密过程
def decrypt(encrypted_with_redundancy, seed, original_order):
    start_time = time.time()  # 开始计时
    encrypted_data = encrypted_with_redundancy[:-8]  # 移除最后8位冗余数据
    print(f"移除冗余后的数据：{encrypted_data}")
    
    dynamic_key = generate_dynamic_key(seed=int(seed))  # 确保解密时使用相同的密钥生成逻辑
    print(f"动态密钥：{dynamic_key}")
    
    decrypted_spliced_data = reverse_dynamic_key(encrypted_data, dynamic_key)
    print(f"使用动态密钥解密后的数据：{decrypted_spliced_data}")
    
    decrypted_data = inverse_splice(decrypted_spliced_data, original_order)
    print(f"逆剪接后的数据：{decrypted_data}")
    
    inverse_substitution_matrix = {v: k for k, v in generate_substitution_matrix(seed).items()}
    decrypted_data = ''.join([inverse_substitution_matrix.get(char, char) for char in decrypted_data])
    print(f"逆转录后的解密数据：{decrypted_data}")
    
    end_time = time.time()  # 结束计时
    decryption_time = end_time - start_time  # 计算解密时间
    print(f"解密时间：{decryption_time:.6f} 秒")
    
    return decrypted_data, decryption_time

# 测试加密解密过程，并计算准确率
def test_encryption_decryption(num_tests):
    seed = "123456789"
    plaintext = "ACGTACGTACGT" * 10  # 增加序列长度
    encryption_times = []
    decryption_times = []

    for _ in range(num_tests):
        encrypted_data, original_order, encryption_time = encrypt(plaintext, seed)
        decrypted_data, decryption_time = decrypt(encrypted_data, seed, original_order)
        
        encryption_times.append(encryption_time)
        decryption_times.append(decryption_time)

    avg_encryption_time = sum(encryption_times) / len(encryption_times)
    avg_decryption_time = sum(decryption_times) / len(decryption_times)

    print(f"平均加密时间：{avg_encryption_time:.6f} 秒")
    print(f"平均解密时间：{avg_decryption_time:.6f} 秒")

    plt.figure(figsize=(10, 5))
    plt.plot(encryption_times, label='Encryption Time')
    plt.plot(decryption_times, label='Decryption Time')
    plt.xlabel('Test Number')
    plt.ylabel('Time (seconds)')
    plt.title('Encryption and Decryption Time per Test')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    num_tests = 10  # 定义测试次数
    test_encryption_decryption(num_tests)