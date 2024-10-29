from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

def encrypt(plaintext, seed, seed_sequence, salt, block_size=190):
    """
    使用RSA加密数据，分块加密，确保每个块的大小小于密钥长度。
    """
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
    private_key = RSA.import_key(private_key_data)
    cipher_rsa = PKCS1_OAEP.new(private_key)

    decrypted_blocks = [
        cipher_rsa.decrypt(block).decode()
        for block in encrypted_data_with_checksum
    ]
    return ''.join(decrypted_blocks)
