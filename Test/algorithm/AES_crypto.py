import os
import threading
import psutil
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
from Crypto.Util.Padding import pad, unpad

# 配置环境变量以限制线程数
def setup_environment():
    """设置单线程环境，确保CPU绑定及线程数为1"""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["BLIS_NUM_THREADS"] = "1"
    p = psutil.Process(os.getpid())
    p.cpu_affinity([0])  # 将进程绑定到单个CPU核心

setup_environment()

# 创建锁对象，确保加密操作是线程安全的
lock = threading.Lock()

def encrypt(plaintext, seed, seed_sequence, salt):
    """
    使用AES加密数据，返回加密后的数据和占位符的substitution_matrix和indices_order。
    """
    with lock:
        key = scrypt(seed.encode(), salt.encode(), 32, N=2**14, r=8, p=1)
        cipher = AES.new(key, AES.MODE_CBC)
        ciphertext = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
        
        # 使用占位符，保持接口一致性
        substitution_matrix = "NA"
        indices_order = "NA"
        
        return cipher.iv + ciphertext, substitution_matrix, indices_order

def decrypt(encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order):
    """
    使用AES解密数据，从加密数据中提取IV，并解密剩余的数据块。
    """
    with lock:
        iv = encrypted_data_with_checksum[:AES.block_size]
        key = scrypt(seed.encode(), salt.encode(), 32, N=2**14, r=8, p=1)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        decrypted_data = unpad(cipher.decrypt(encrypted_data_with_checksum[AES.block_size:]), AES.block_size).decode()
        return decrypted_data
