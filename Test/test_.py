import ctypes
import psutil
import threading
import os

# 1. 加载C库
# 对于Linux/Mac，使用CDLL加载共享库（.so）
# crypto_lib = ctypes.CDLL("/path/to/your/libcryptodome.so")

# 对于Windows，使用windll加载动态链接库（.dll）
crypto_lib = ctypes.windll.LoadLibrary("C:/path/to/your/cryptodome.dll")

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

# 2. 实时检测线程数量
def check_threading_during_execution(name):
    """检查当前活动的线程数，确保在执行加密/解密时保持单线程"""
    process = psutil.Process()
    num_threads = process.num_threads()
    if num_threads > 1:
        print(f"[{name}] Warning: Multiple threads detected during execution! Number of threads: {num_threads}")
    else:
        print(f"[{name}] Execution is single-threaded. Number of threads: {num_threads}")

# 3. 定义加密函数的参数和返回类型
# 假设C函数 rsa_encrypt 接受一个字符数组指针和长度，并返回加密后的结果
crypto_lib.rsa_encrypt.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
crypto_lib.rsa_encrypt.restype = ctypes.POINTER(ctypes.c_char)

# 定义解密函数的参数和返回类型
crypto_lib.rsa_decrypt.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
crypto_lib.rsa_decrypt.restype = ctypes.POINTER(ctypes.c_char)

# 加密和解密监控类
class CryptoMonitor:
    def __init__(self, name):
        self.name = name

    def rsa_encrypt(self, plaintext):
        """调用C库的RSA加密函数，并检测执行过程中的线程数"""
        plaintext_len = len(plaintext)
        plaintext_ptr = ctypes.create_string_buffer(plaintext.encode('utf-8'))
        plaintext_ptr_cast = ctypes.cast(plaintext_ptr, ctypes.POINTER(ctypes.c_char))

        # 检查加密过程中是否有多个线程
        check_threading_during_execution(self.name + " - RSA Encrypt")

        # 调用C库中的加密函数
        encrypted_ptr = crypto_lib.rsa_encrypt(plaintext_ptr_cast, plaintext_len)
        encrypted_message = ctypes.string_at(encrypted_ptr)

        return encrypted_message

    def rsa_decrypt(self, encrypted_message):
        """调用C库的RSA解密函数，并检测执行过程中的线程数"""
        encrypted_len = len(encrypted_message)
        encrypted_ptr_cast = ctypes.cast(ctypes.create_string_buffer(encrypted_message), ctypes.POINTER(ctypes.c_char))

        # 检查解密过程中是否有多个线程
        check_threading_during_execution(self.name + " - RSA Decrypt")

        # 调用C库中的解密函数
        decrypted_ptr = crypto_lib.rsa_decrypt(encrypted_ptr_cast, encrypted_len)
        decrypted_message = ctypes.string_at(decrypted_ptr).decode('utf-8')

        return decrypted_message

# 测试加密和解密操作
def test_crypto_operations():
    crypto_monitor = CryptoMonitor(name="CryptoAlgorithm")

    # 加密数据
    plaintext = "Hello, RSA!"
    print("Encrypting data...")
    encrypted_data = crypto_monitor.rsa_encrypt(plaintext)
    print(f"Encrypted data: {encrypted_data}")

    # 解密数据
    print("Decrypting data...")
    decrypted_data = crypto_monitor.rsa_decrypt(encrypted_data)
    print(f"Decrypted data: {decrypted_data}")

# 测试主函数，启动加密操作线程并强制停止
def main():
    # 启动加密操作线程
    t = threading.Thread(target=test_crypto_operations)
    t.start()

    # 让线程运行5秒
    t.join(5)

    print("Main thread finished.")

if __name__ == "__main__":
    main()
