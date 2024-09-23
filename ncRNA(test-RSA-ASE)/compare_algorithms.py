import matplotlib.pyplot as plt
import time
import random
import hashlib
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import numpy as np

# -------------------------------------------
# Custom Encryption Functions
# -------------------------------------------
def generate_substitution_matrix(seed):
    random.seed(seed)
    bases = ['A', 'C', 'G', 'T']
    substitution = dict()
    shuffled_bases = random.sample(bases, len(bases))
    for i, base in enumerate(bases):
        substitution[base] = shuffled_bases[i]
    return substitution

def transcribe_dna_to_rna(plaintext, substitution_matrix):
    transcribed = ''.join([substitution_matrix.get(char, char) for char in plaintext])
    return transcribed

def split_and_splice(data, block_size=3):
    blocks = [data[i:i+block_size] for i in range(0, len(data), block_size)]
    original_order = blocks.copy()
    random.shuffle(blocks)
    spliced_data = ''.join(blocks)
    return spliced_data, original_order

def inverse_splice(data, original_order):
    return ''.join(original_order)

def generate_dynamic_key(seed=None):
    if seed is None:
        seed = int(time.time())
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

def insert_redundancy(encrypted_data):
    redundancy = ''.join(random.choice('01') for _ in range(8))
    return encrypted_data + redundancy

# 自定义加密函数
def custom_encrypt(plaintext, seed=None):
    start_time = time.time()
    substitution_matrix = generate_substitution_matrix(seed)
    transcribed_data = transcribe_dna_to_rna(plaintext, substitution_matrix)
    spliced_data, original_order = split_and_splice(transcribed_data)
    dynamic_key = generate_dynamic_key(seed)
    encrypted_data = apply_dynamic_key(spliced_data, dynamic_key)
    encrypted_with_redundancy = insert_redundancy(encrypted_data)
    encryption_time = time.time() - start_time
    print(f"Encrypted data (first 50 chars): {encrypted_with_redundancy[:50]}")  # 添加这行来检查加密后的数据
    return encrypted_with_redundancy, original_order, encryption_time

# 自定义解密函数
def custom_decrypt(encrypted_with_redundancy, seed, original_order):
    start_time = time.time()
    encrypted_data = encrypted_with_redundancy[:-8]  # Remove redundancy
    dynamic_key = generate_dynamic_key(seed)
    decrypted_spliced_data = reverse_dynamic_key(encrypted_data, dynamic_key)
    decrypted_data = inverse_splice(decrypted_spliced_data, original_order)
    inverse_substitution_matrix = {v: k for k, v in generate_substitution_matrix(seed).items()}
    recovered_plaintext = ''.join([inverse_substitution_matrix.get(char, char) for char in decrypted_data])
    decryption_time = time.time() - start_time
    return recovered_plaintext, decryption_time

# -------------------------------------------
# RSA Encryption
# -------------------------------------------
def generate_rsa_keys():
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return private_key, public_key

def rsa_encrypt(public_key, data):
    rsa_key = RSA.import_key(public_key)
    cipher_rsa = PKCS1_OAEP.new(rsa_key)
    encrypted = cipher_rsa.encrypt(data.encode())
    return encrypted

def rsa_decrypt(private_key, data):
    rsa_key = RSA.import_key(private_key)
    cipher_rsa = PKCS1_OAEP.new(rsa_key)
    decrypted = cipher_rsa.decrypt(data).decode()
    return decrypted

# -------------------------------------------
# AES Encryption
# -------------------------------------------
def aes_encrypt(data):
    key = get_random_bytes(16)
    cipher_aes = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher_aes.encrypt(pad(data.encode(), AES.block_size))
    return cipher_aes.iv + ct_bytes, key

def aes_decrypt(key, data):
    iv = data[:AES.block_size]
    cipher_aes = AES.new(key, AES.MODE_CBC, iv)
    decrypted = unpad(cipher_aes.decrypt(data[AES.block_size:]), AES.block_size).decode()
    return decrypted

# -------------------------------------------
# Performance Metrics
# -------------------------------------------
def calculate_accuracy(original, decrypted):
    return sum([1 if o == d else 0 for o, d in zip(original, decrypted)]) / len(original)

def calculate_entropy(data):
    if isinstance(data, str):
        data_bytes = bytes(data, 'utf-8')  # 使用 utf-8 而不是 latin1
    else:
        data_bytes = data
    value, counts = np.unique(list(data_bytes), return_counts=True)
    probabilities = counts / len(data_bytes)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# -------------------------------------------
# Test and Compare Functions
# -------------------------------------------
def generate_complex_dataset(num_sequences=10, seq_length=1000):
    bases = ['A', 'C', 'G', 'T']
    dataset = []
    for _ in range(num_sequences):
        sequence = ''.join(random.choice(bases) for _ in range(seq_length))
        dataset.append(sequence)
    return dataset

def generate_simple_dataset(num_sequences=10, seq_length=120):
    bases = ['A', 'C', 'G', 'T']
    dataset = []
    for _ in range(num_sequences):
        sequence = ''.join(random.choice(bases) for _ in range(seq_length))
        dataset.append(sequence)
    return dataset

def test_custom_encryption(dataset, num_tests=10):
    seed = "123456789"
    encryption_times = []
    decryption_times = []
    accuracies = []
    entropies = []

    for data in dataset:
        encrypted_data, original_order, encryption_time = custom_encrypt(data, seed)
        encryption_times.append(encryption_time)

        decrypted_data, decryption_time = custom_decrypt(encrypted_data, seed, original_order)
        decryption_times.append(decryption_time)

        accuracy = calculate_accuracy(data, decrypted_data)
        accuracies.append(accuracy)

        entropy = calculate_entropy(encrypted_data)
        entropies.append(entropy)

    print(f"自定义算法 - 平均加密时间：{sum(encryption_times)/num_tests:.6f} 秒")
    print(f"自定义算法 - 平均解密时间：{sum(decryption_times)/num_tests:.6f} 秒")
    print(f"自定义算法 - 平均准确度：{sum(accuracies)/num_tests:.2%}")
    print(f"自定义算法 - 平均熵：{sum(entropies)/num_tests:.4f}")

    return encryption_times, decryption_times, accuracies, entropies

def test_rsa_encryption(dataset, num_tests=10):
    private_key, public_key = generate_rsa_keys()
    encryption_times = []
    decryption_times = []
    accuracies = []
    entropies = []

    for data in dataset:
        start_time = time.time()
        encrypted_data = rsa_encrypt(public_key, data[:128])  # 限制数据长度
        encryption_times.append(time.time() - start_time)

        start_time = time.time()
        decrypted_data = rsa_decrypt(private_key, encrypted_data)
        decryption_times.append(time.time() - start_time)

        accuracy = calculate_accuracy(data[:128], decrypted_data)
        accuracies.append(accuracy)

        entropy = calculate_entropy(encrypted_data)
        entropies.append(entropy)

    print(f"RSA算法 - 平均加密时间：{sum(encryption_times)/num_tests:.6f} 秒")
    print(f"RSA算法 - 平均解密时间：{sum(decryption_times)/num_tests:.6f} 秒")
    print(f"RSA算法 - 平均准确度：{sum(accuracies)/num_tests:.2%}")
    print(f"RSA算法 - 平均熵：{sum(entropies)/num_tests:.4f}")

    return encryption_times, decryption_times, accuracies, entropies

def test_aes_encryption(dataset, num_tests=10):
    encryption_times = []
    decryption_times = []
    accuracies = []
    entropies = []

    for data in dataset:
        start_time = time.time()
        encrypted_data, aes_key = aes_encrypt(data)
        encryption_times.append(time.time() - start_time)

        start_time = time.time()
        decrypted_data = aes_decrypt(aes_key, encrypted_data)
        decryption_times.append(time.time() - start_time)

        accuracy = calculate_accuracy(data, decrypted_data)
        accuracies.append(accuracy)

        entropy = calculate_entropy(encrypted_data)
        entropies.append(entropy)

    print(f"AES算法 - 平均加密时间：{sum(encryption_times)/num_tests:.6f} 秒")
    print(f"AES算法 - 平均解密时间：{sum(decryption_times)/num_tests:.6f} 秒")
    print(f"AES算法 - 平均准确度：{sum(accuracies)/num_tests:.2%}")
    print(f"AES算法 - 平均熵：{sum(entropies)/num_tests:.4f}")

    return encryption_times, decryption_times, accuracies, entropies

# -------------------------------------------
# Run Comparison
# -------------------------------------------
def run_comparison():
    num_tests = 10

    # 生成两组数据集
    complex_dataset = generate_complex_dataset(num_tests)
    simple_dataset = generate_simple_dataset(num_tests)

    # 测试复杂数据集
    print("测试复杂数据集:")
    custom_enc_times_complex, custom_dec_times_complex, custom_accuracies_complex, custom_entropies_complex = test_custom_encryption(complex_dataset, num_tests)
    rsa_enc_times_complex, rsa_dec_times_complex, rsa_accuracies_complex, rsa_entropies_complex = test_rsa_encryption(complex_dataset, num_tests)
    aes_enc_times_complex, aes_dec_times_complex, aes_accuracies_complex, aes_entropies_complex = test_aes_encryption(complex_dataset, num_tests)

    # 测试简单数据集
    print("\n测试简单数据集:")
    custom_enc_times_simple, custom_dec_times_simple, custom_accuracies_simple, custom_entropies_simple = test_custom_encryption(simple_dataset, num_tests)
    rsa_enc_times_simple, rsa_dec_times_simple, rsa_accuracies_simple, rsa_entropies_simple = test_rsa_encryption(simple_dataset, num_tests)
    aes_enc_times_simple, aes_dec_times_simple, aes_accuracies_simple, aes_entropies_simple = test_aes_encryption(simple_dataset, num_tests)

    # 创建4个并行图表
    fig, axs = plt.subplots(4, 2, figsize=(20, 30))
    
    # 加密时间比较
    axs[0, 0].set_title('Encryption Time Comparison (Complex Dataset)')
    axs[0, 0].plot(custom_enc_times_complex, 'o-', label='Custom')
    axs[0, 0].plot(rsa_enc_times_complex, 's-', label='RSA')
    axs[0, 0].plot(aes_enc_times_complex, '^-', label='AES')
    axs[0, 0].set_xlabel('Test Number')
    axs[0, 0].set_ylabel('Time (seconds)')
    axs[0, 0].legend()

    axs[0, 1].set_title('Encryption Time Comparison (Simple Dataset)')
    axs[0, 1].plot(custom_enc_times_simple, 'o-', label='Custom')
    axs[0, 1].plot(rsa_enc_times_simple, 's-', label='RSA')
    axs[0, 1].plot(aes_enc_times_simple, '^-', label='AES')
    axs[0, 1].set_xlabel('Test Number')
    axs[0, 1].set_ylabel('Time (seconds)')
    axs[0, 1].legend()

    # 解密时间比较
    axs[1, 0].set_title('Decryption Time Comparison (Complex Dataset)')
    axs[1, 0].plot(custom_dec_times_complex, 'o-', label='Custom')
    axs[1, 0].plot(rsa_dec_times_complex, 's-', label='RSA')
    axs[1, 0].plot(aes_dec_times_complex, '^-', label='AES')
    axs[1, 0].set_xlabel('Test Number')
    axs[1, 0].set_ylabel('Time (seconds)')
    axs[1, 0].legend()

    axs[1, 1].set_title('Decryption Time Comparison (Simple Dataset)')
    axs[1, 1].plot(custom_dec_times_simple, 'o-', label='Custom')
    axs[1, 1].plot(rsa_dec_times_simple, 's-', label='RSA')
    axs[1, 1].plot(aes_dec_times_simple, '^-', label='AES')
    axs[1, 1].set_xlabel('Test Number')
    axs[1, 1].set_ylabel('Time (seconds)')
    axs[1, 1].legend()

    # 准确度比较
    axs[2, 0].set_title('Accuracy Comparison (Complex Dataset)')
    axs[2, 0].plot(custom_accuracies_complex, 'o-', label='Custom')
    axs[2, 0].plot(rsa_accuracies_complex, 's-', label='RSA')
    axs[2, 0].plot(aes_accuracies_complex, '^-', label='AES')
    axs[2, 0].set_xlabel('Test Number')
    axs[2, 0].set_ylabel('Accuracy')
    axs[2, 0].legend()

    axs[2, 1].set_title('Accuracy Comparison (Simple Dataset)')
    axs[2, 1].plot(custom_accuracies_simple, 'o-', label='Custom')
    axs[2, 1].plot(rsa_accuracies_simple, 's-', label='RSA')
    axs[2, 1].plot(aes_accuracies_simple, '^-', label='AES')
    axs[2, 1].set_xlabel('Test Number')
    axs[2, 1].set_ylabel('Accuracy')
    axs[2, 1].legend()

    # 熵比较
    axs[3, 0].set_title('Entropy Comparison (Complex Dataset)')
    axs[3, 0].plot(custom_entropies_complex, 'o-', label='Custom')
    axs[3, 0].plot(rsa_entropies_complex, 's-', label='RSA')
    axs[3, 0].plot(aes_entropies_complex, '^-', label='AES')
    axs[3, 0].set_xlabel('Test Number')
    axs[3, 0].set_ylabel('Entropy')
    axs[3, 0].legend()

    axs[3, 1].set_title('Entropy Comparison (Simple Dataset)')
    axs[3, 1].plot(custom_entropies_simple, 'o-', label='Custom')
    axs[3, 1].plot(rsa_entropies_simple, 's-', label='RSA')
    axs[3, 1].plot(aes_entropies_simple, '^-', label='AES')
    axs[3, 1].set_xlabel('Test Number')
    axs[3, 1].set_ylabel('Entropy')
    axs[3, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()
