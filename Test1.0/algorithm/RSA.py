from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

# 加密函数
def rsa_encrypt(plaintext, block_size=190):
    """
    使用动态生成的公钥对数据进行分块加密，确保每个块的大小小于密钥长度。
    
    参数：
        plaintext (str): 需要加密的明文。
        block_size (int): 每次加密的块大小，默认值为190。
    返回：
        Tuple[List[bytes], bytes, bytes]: 每个块加密后的数据组成的列表，公钥和私钥。
    """
    key = RSA.generate(2048)
    public_key = key.publickey()
    cipher_rsa = PKCS1_OAEP.new(public_key)

    encrypted_blocks = [
        cipher_rsa.encrypt(plaintext[i:i + block_size].encode())
        for i in range(0, len(plaintext), block_size)
    ]
    return encrypted_blocks, public_key.export_key(), key.export_key()

# 解密函数
def rsa_decrypt(encrypted_data, private_key_data):
    """
    使用私钥对数据进行解密，逐块解密后重新组合成明文。
    
    参数：
        encrypted_data (List[bytes]): 分块加密后的数据列表。
        private_key_data (bytes): 私钥数据。
    返回：
        str: 解密后的明文。
    """
    private_key = RSA.import_key(private_key_data)
    cipher_rsa = PKCS1_OAEP.new(private_key)

    decrypted_blocks = [
        cipher_rsa.decrypt(block).decode()
        for block in encrypted_data
    ]
    return ''.join(decrypted_blocks)