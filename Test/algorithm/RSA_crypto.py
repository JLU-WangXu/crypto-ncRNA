import os
import threading
import psutil
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

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

def encrypt(plaintext, seed, seed_sequence, salt, block_size=190):
    """
    使用RSA加密数据，分块加密，确保每个块的大小小于密钥长度。
    """
    with lock:
        key = RSA.generate(2048)
        public_key = key.publickey()
        cipher_rsa = PKCS1_OAEP.new(public_key)

        encrypted_blocks = [
            cipher_rsa.encrypt(plaintext[i:i + block_size].encode())
            for i in range(0, len(plaintext), block_size)
        ]
        return encrypted_blocks, public_key.export_key(), key.export_key()

def decrypt(encrypted_data_with_checksum, seed, seed_sequence, salt, public_key_data, private_key_data):
    """
    使用RSA私钥解密数据，逐块解密后重新组合成明文。
    """
    with lock:
        private_key = RSA.import_key(private_key_data)
        cipher_rsa = PKCS1_OAEP.new(private_key)

        decrypted_blocks = [
            cipher_rsa.decrypt(block).decode()
            for block in encrypted_data_with_checksum
        ]
        return ''.join(decrypted_blocks)
