# crypto-ncRNA: 基于非编码RNA（ncRNA）的加密算法项目

**crypto-ncRNA** 是一个基于非编码RNA（ncRNA）的生物启发式加密算法项目，结合生物序列特性和现代密码学技术，旨在研究ncRNA在信息加密与数据保护中的潜力。该项目模拟ncRNA的动态行为，开发出一套独特的加密系统，通过基因序列转录、动态密钥生成、冗余保护等机制，实现对文本、基因数据等多种数据类型的加密与解密。

祝贺！该项目目前已被 ICLR 2025 Workshop on AI for Nucleic Acids 接收！  2025-03-05

## Author: 

Xu Wang<sup>1, 4, a*</sup>, Yiquan Wang<sup>2, 4, b</sup>, Tin-Yeh Huang<sup>3, 4, c</sup>

1. Tsinghua University-Peking University Joint Center for Life Sciences, Tsinghua University, Beijing, 100084, China

2. College of Mathematics and System Science, Xinjiang University, Urumqi, Xinjiang, 830046, China

3. Department of Industrial and System Engineering, Faculty of Engineering, The Hong Kong Polytechnic University, Hong Kong SAR, 999077, China

4. Shenzhen X-Institute, Shenzhen, China, 518055

a. [wangxu24@mails.tsinghua.edu.cn](mailto:wangxu24@mails.tsinghua.edu.cn)   b. [ethan@stu.xju.edu.cn](mailto:ethan@stu.xju.edu.cn)   c. [tin-yeh.huang@connect.polyu.hk](mailto:tin-yeh.huang@connect.polyu.hk)  

## 引用
如果您使用本项目或参考相关算法，请使用以下引用格式：
```bibtex
@inproceedings{
wangxu2025cryptoncrna,
title={Crypto-nc{RNA}: Non-coding {RNA} (nc{RNA}) Based Encryption Algorithm},
author={WangXu and YiquanWang and Tin-Yeh HUANG},
booktitle={ICLR 2025 Workshop on AI for Nucleic Acids},
year={2025},
url={https://openreview.net/forum?id=j6ODUDw4vN}
}
```

### 1.0 版本亮点：
1. **基于ncRNA的加密算法**：利用碱基替换与RNA转录模拟加密。
2. **动态密钥生成**：使用输入属性或时间动态生成加密密钥。
3. **冗余保护**：加入冗余位增强数据完整性与抗攻击性。
4. **加密解密功能**：提供完整的加密解密流程，支持文本与基因数据的加密。

### 2.0 版本改进：
1. **替换矩阵复杂度提升**：将碱基替换扩展至密码子（3个核苷酸），大幅增加替换方式的可能性，增强抗攻击性。
2. **引入非线性变换**：通过RNA二级结构折叠模拟非线性加密过程，提升抵抗线性攻击的能力。
3. **动态密钥安全性增强**：采用生物序列生成高熵密钥，保证密钥难以预测。
4. **加入生物纠错机制**：通过冗余校验码提升数据完整性和抗干扰性。

### 3.0 版本改进：
1. **Nussinov算法集成**：使用Nussinov算法模拟RNA二级结构折叠，进一步增强加密的非线性和不可预测性。
2. **使用实际生物序列作为密钥种子**：引入实际的RNA序列作为密钥种子，增加密钥生成的生物学意义和不可预测性。
3. **性能优化**：改进算法结构，提高加密解密的效率，特别是在处理大数据集时。

通过这些改进，**crypto-ncRNA** 项目不仅保持了生物学启发的核心理念，还大大增强了算法的安全性和复杂性，使其能够有效抵抗穷举攻击、线性攻击等已知的加密破解手段。

# 目录部分
# 1.0版本

- [背景](#背景)
- [功能](#功能)
- [依赖](#依赖)
- [安装](#安装)
- [使用](#使用)
  - [加密](#加密)
  - [解密](#解密)
  - [示例](#示例)
- [测试](#测试)
- [结果分析](#结果分析)
- [许可证](#许可证)

# 2.0版本

- [算法改进方案](#算法改进方案)
  - [增加替换矩阵的复杂度](#增加替换矩阵的复杂度)
  - [引入生物学中的非线性过程](#引入生物学中的非线性过程)
  - [动态密钥的改进](#动态密钥的改进)
  - [加入生物学中的冗余和纠错机制](#加入生物学中的冗余和纠错机制)
  
- [数学分析](#数学分析)
  - [替换矩阵的安全性分析](#替换矩阵的安全性分析)
  - [非线性变换的安全性分析](#非线性变换的安全性分析)
  - [动态密钥的安全性分析](#动态密钥的安全性分析)
  - [整体安全性分析](#整体安全性分析)

- [加密和解密算法](#加密和解密算法)
  - [加密过程](#加密过程)
  - [解密过程](#解密过程)

- [数学证明](#数学证明)
- [抗攻击性分析](#抗攻击性分析)

# 3.0版本

- [改进内容概述](#改进内容概述)
  - [实现Nussinov算法](#实现nussinov算法)
  - [使用实际生物序列作为密钥种子](#使用实际生物序列作为密钥种子)
  - [性能优化](#性能优化)
  
- [详细实现](#详细实现)
  - [使用Nussinov算法模拟RNA二级结构折叠](#使用nussinov算法模拟rna二级结构折叠)
  - [加密过程中的Nussinov算法集成](#加密过程中的nussinov算法集成)
  - [解密过程中的逆Nussinov算法](#解密过程中的逆nussinov算法)


------------------------------------------------------------------------------------------------------------------------------------------------------------------


# 1.0版本主体


## 背景

非编码RNA（ncRNA）在生物体内具有重要的调控作用。它们不仅在基因表达的调控中发挥作用，还展示了高度复杂的序列模式。**crypto-ncRNA**项目通过模拟这些生物序列的动态行为，开发了一种独特的信息加密方式。该项目结合生物序列和现代加密算法，旨在创建一个既具有理论意义又具有实际应用价值的加密系统。




<div align="center">
  <img src="https://github.com/JLU-WangXu/crypto-ncRNA/blob/main/pic/ncRNA.png" alt="ncRNA图示" width="400"/>
</div>



## 功能

- **基于ncRNA的加密算法**：利用非编码RNA的特性进行加密，数据被转换为模拟的RNA序列，并通过自定义加密过程实现信息保护。

 ```python
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
 ```


- **动态密钥生成**：采用动态生成的密钥进行加密，密钥基于输入数据的特定属性或时间生成。
  
 ```python
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
 ```

- **冗余保护**：在加密数据中加入冗余位以增强其完整性和抗攻击性。
  
 ```python
  def insert_redundancy(encrypted_data):
    redundancy = ''.join(random.choice('01') for _ in range(8))  # 8位随机冗余
    return encrypted_data + redundancy
 ```

- **加密与解密功能**：实现了完整的加密和解密过程，可以应用于文本、基因数据等多种数据类型。
  
 ## 加密 

 ```python
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

 ```


## 解密
```python
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
```


## 依赖

为了运行该项目，您需要以下依赖项：

- Python 3.x
- `pycryptodome`：用于加密和解密功能的库
- `numpy`：用于数据处理
- `matplotlib`：用于结果的可视化（可选）

通过以下命令安装依赖项：

```bash
pip install pycryptodome numpy matplotlib
```

## 安装

1. 克隆项目到本地：

    ```bash
    git clone https://github.com/JLU-WangXu/cryto-ncRNA.git
    cd cryto-ncRNA
    ```

2. 安装所有必要的依赖项（如上所示）：

    ```bash
    pip install pycryptodome numpy matplotlib
    ```

3. 确保你使用的是Python 3.x版本。

## 使用

### 加密

要进行加密，可以使用 `encrypt()` 函数，该函数接受文本或基因序列作为输入，并通过自定义加密算法返回加密后的数据。

```python
from encryption_algorithms import encrypt

# 输入数据
plaintext = "ACGTACGTACGT"

# 加密数据
encrypted_data, original_order, encryption_time = encrypt(plaintext, seed="123456789")

print(f"Encrypted Data: {encrypted_data}")
```
## 解密

要解密加密的数据，可以使用 `decrypt()` 函数，确保使用与加密时相同的动态密钥和顺序。

```python
from encryption_algorithms import decrypt

# 解密数据
decrypted_data, decryption_time = decrypt(encrypted_data, seed="123456789", original_order=original_order)

print(f"Decrypted Data: {decrypted_data}")
```

### 示例

可以通过 main 函数测试加密解密流程，并验证解密的准确性：

```python

if __name__ == "__main__":
    plaintext = "ACGTACGTACGT"

    # 加密
    encrypted_data, original_order, encryption_time = encrypt(plaintext, seed="123456789")
    
    # 解密
    decrypted_data, decryption_time = decrypt(encrypted_data, seed="123456789", original_order=original_order)
    
    # 准确性验证
    if plaintext == decrypted_data:
        accuracy = 100.0
    else:
        accuracy = 0.0
    print(f"Encryption Accuracy: {accuracy}%")
    print(f"Encryption Time: {encryption_time:.6f} seconds")
    print(f"Decryption Time: {decryption_time:.6f} seconds")
```

## 测试
可以使用 test_algorithms.py 文件中定义的函数对不同类型的数据集（文本、基因数据）进行加密和解密测试。通过运行此脚本，你可以获得加密时间、解密时间、熵值等结果。

运行以下命令启动测试：
```python
python test_algorithms.py
```

## 结果分析
算法的性能和准确性将在控制台输出，并包括以下信息：

加密时间：表示加密过程的运行时间。

解密时间：表示解密过程的运行时间。

准确性：解密后数据与原始数据的匹配度。

熵值：用于评估加密后数据的随机性。

## 许可证
此项目根据MIT许可证开源，详情请参阅 LICENSE 文件。


# 2.0改进版本：

# 一、算法改进方案

## 1. 增加替换矩阵的复杂度

问题分析：

- 原始替换矩阵的弱点： 仅使用4个碱基（'A', 'C', 'G', 'T'）进行替换，可能的替换方式只有4! = 24种，容易被穷举攻击破解。

改进方案：

- 扩展碱基集合： 引入RNA序列中的核苷酸（包括尿嘧啶'U'）或考虑密码子（三联体）编码。

- 使用密码子替换：

- 密码子数量： 有4^3 = 64种可能的密码子（三个核苷酸组成一个密码子）。
  
- 替换矩阵大小： 使用密码子替换可以将替换矩阵的可能性增加到64!，大大增加了穷举破解的难度。

实现方式：

- 明文编码： 将明文字符映射到密码子上，每个字符对应一个或多个密码子。

- 替换矩阵生成： 基于种子生成一个随机的密码子替换矩阵，确保替换方式的不可预测性。

## 2. 引入生物学中的非线性过程

问题分析：

- 异或操作的线性性： 异或操作是线性的，缺乏密码学中的混淆和扩散特性。

改进方案：

- 模拟RNA二级结构折叠： RNA分子可以形成复杂的二级结构，如发卡结构、茎环结构等。这些结构形成的过程是非线性的。

- 引入非线性函数：

- 函数定义： 定义一个基于RNA二级结构预测算法的非线性函数f，将序列映射到其二级结构表示。

- 加密过程： 在转录和剪接之后，对序列应用函数f，引入非线性变换。

实现方式：

- 使用RNA二级结构预测算法： 如Nussinov算法或能量最小化方法，计算序列的二级结构。

- 将结构信息用于加密： 将二级结构的表示（如配对信息、能量值）引入到加密过程中，增加复杂性。

## 3. 动态密钥的改进

问题分析：

- 种子的安全性： 如果种子易于预测，动态密钥的安全性会受到威胁。

改进方案：

- 基于生物序列的种子： 使用难以预测的生物序列（如基因序列、蛋白质序列）作为种子。

- 增加种子的熵： 将种子扩展为高熵的生物数据，并通过哈希函数生成动态密钥。

实现方式：

- 种子生成：

- 选择生物序列： 从公共数据库中选择特定的生物序列，或者使用双方共享的私有生物数据。

- 哈希处理： 对生物序列进行哈希（如SHA-256），生成高熵的动态密钥。

## 4. 加入生物学中的冗余和纠错机制

问题分析：

- 冗余位的简单性： 原有的冗余位不能有效提供数据完整性验证。

改进方案：

- 模拟生物纠错机制： 生物系统中存在如DNA修复、校对等机制，确保遗传信息的准确性。

- 引入校验和或纠错码： 使用如CRC校验、哈夫曼编码等，提高数据完整性。

实现方式：

- 数据完整性验证：

- 计算校验和： 在加密数据中加入基于生物序列的校验和。

- 纠错编码： 使用纠错码，如汉明码，增加数据的冗余度，增强抗干扰能力。


---

# 二、数学分析

## 1. 替换矩阵的安全性分析

增加替换矩阵的复杂度：

- 密码子替换矩阵数量： 64! ≈ 1.27 x 10^89。

- 穷举破解的难度： 由于可能的替换方式数量极大，穷举所有替换矩阵在计算上不可行。

结论：

- 安全性提升： 替换矩阵的空间大幅增加，抵抗穷举攻击的能力显著增强。

## 2. 非线性变换的安全性分析

引入RNA二级结构的非线性函数：

- 函数复杂度： RNA二级结构预测是一个NP完全问题，具有高度的计算复杂度。

- 逆向困难性： 在未知函数f和参数的情况下，攻击者很难从输出推导出输入。

结论：

- 抵抗线性攻击： 非线性变换增加了算法的混淆和扩散特性，提高了抵抗线性分析攻击的能力。

## 3. 动态密钥的安全性分析

使用高熵种子生成动态密钥：

- 种子熵的增加： 生物序列具有高复杂度和随机性，作为种子可以提供高熵的输入。

- 密钥空间： 动态密钥长度为256位，密钥空间为2^256。

结论：

- 抵抗种子预测攻击： 高熵的生物序列使得种子难以预测，动态密钥的安全性得到保障。

## 4. 整体安全性分析

抵抗已知明文攻击：

- 复杂的替换矩阵和非线性变换： 使得从已知明文和密文对推导密钥变得困难。

抵抗频率分析攻击：

- 替换矩阵的多样性： 扩大的字符集和替换方式模糊了字符频率分布。

抵抗选择明文攻击：

- 非线性和高熵密钥： 即使攻击者可以选择明文进行加密，也难以推导出密钥或逆向非线性函数。


---

# 三、算法的详细描述

## 1. 加密过程

```python
def encrypt(plaintext, seed=None):
    # Step 1: 明文编码为密码子序列
    codon_sequence = encode_plaintext_to_codons(plaintext)
    
    # Step 2: 生成密码子替换矩阵
    substitution_matrix = generate_codon_substitution_matrix(seed)
    
    # Step 3: 替换密码子序列
    substituted_sequence = substitute_codons(codon_sequence, substitution_matrix)
    
    # Step 4: 应用RNA二级结构非线性变换
    nonlinear_transformed_sequence = apply_rna_secondary_structure(substituted_sequence)
    
    # Step 5: 基于生物序列生成动态密钥
    dynamic_key = generate_dynamic_key_from_biological_data(seed)
    
    # Step 6: 使用动态密钥加密（如AES）
    encrypted_data = aes_encrypt(nonlinear_transformed_sequence, dynamic_key)
    
    # Step 7: 添加纠错码或校验和
    encrypted_data_with_ecc = add_error_correction_code(encrypted_data)
    
    return encrypted_data_with_ecc
````

## 2. 解密过程

```python
def decrypt(encrypted_data_with_ecc, seed):
    # Step 1: 验证并移除纠错码或校验和
    encrypted_data = verify_and_remove_ecc(encrypted_data_with_ecc)
    
    # Step 2: 基于生物序列生成动态密钥
    dynamic_key = generate_dynamic_key_from_biological_data(seed)
    
    # Step 3: 使用动态密钥解密（如AES）
    decrypted_sequence = aes_decrypt(encrypted_data, dynamic_key)
    
    # Step 4: 逆应用RNA二级结构非线性变换
    inverse_nonlinear_sequence = inverse_rna_secondary_structure(decrypted_sequence)
    
    # Step 5: 逆替换密码子序列
    original_codon_sequence = inverse_substitute_codons(inverse_nonlinear_sequence, substitution_matrix)
    
    # Step 6: 将密码子序列解码为明文
    plaintext = decode_codons_to_plaintext(original_codon_sequence)
    
    return plaintext
```
---

# 四、具体的数学证明

## 1. 密钥空间大小的计算

密码子替换矩阵数量：

- 总数： 64!。

- 计算： 64! ≈ 1.27 x 10^89。

结论：

- 穷举破解不可行： 即使每秒尝试10^18种替换方式，耗时仍然超过10^71年。

## 2. 非线性函数的不可逆性

RNA二级结构预测的计算复杂度：

- 问题性质： RNA二级结构预测是NP完全问题。

- 逆向困难性： 由于计算复杂度高，逆向求解原始序列在计算上不可行。

结论：

- 安全性保证： 非线性变换增加了算法的不可逆性，增强了安全性。

## 3. 动态密钥的安全性

基于哈希函数的密钥生成：

- 哈希函数性质： SHA-256等哈希函数具有抗碰撞和单向性。

- 密钥空间： 2^256，足以抵抗当前和可预见未来的暴力破解。

结论：

- 密钥安全性高： 只要种子安全，动态密钥在计算上不可逆，安全性得到保障。

## 4. 抗攻击性分析

已知明文攻击：

- 难点： 即使攻击者知道部分明文和对应的密文，由于非线性变换和高熵密钥，无法推导出密钥或逆向变换。

频率分析攻击：

- 难点： 扩大的替换矩阵和非线性变换模糊了字符频率，频率分析失效。

选择明文攻击：

- 难点： 非线性变换和动态密钥的引入，使得攻击者无法利用选择的明文推导密钥或算法结构。


---

# 五、总结

通过以上改进，我们在保留生物学启发特性的基础上，增强了算法的安全性，并提供了详细的数学分析。

- 增加替换矩阵的复杂度： 使用密码子替换，大幅扩展了替换方式的可能性。

- 引入非线性变换： 模拟RNA二级结构折叠，增加算法的非线性和不可逆性。

- 增强动态密钥生成： 使用高熵的生物序列作为种子，确保动态密钥的安全性。

- 加入数据完整性验证： 使用纠错码或校验和，提供数据完整性和抗干扰能力。

- 数学证明： 通过计算密钥空间、分析算法复杂度，证明了算法能够抵抗已知的攻击。

最终结果：

- 算法安全性得到显著提高： 能够抵抗穷举攻击、频率分析、已知明文攻击等。

- 保留生物学特性： 算法各个步骤都以生物过程为基础，保持了生物启发的核心理念。




# 3.0改进版本

使用 Nussinov 算法模拟 RNA 二级结构折叠，以增加非线性变换的复杂度。
使用实际的生物序列作为密钥种子，提高算法的生物学意义和安全性。

# 一、改进内容概述
## 1. 实现 Nussinov 算法
目的：增加非线性变换的复杂度，增强算法的安全性。
实现：将 Nussinov 算法集成到加密过程中的 RNA 二级结构折叠步骤。
## 2. 使用实际的生物序列作为密钥种子
目的：增强密钥的不可预测性和生物学意义。
实现：从公共生物数据库（如 NCBI）获取实际的 RNA 序列，或使用用户提供的生物序列。
## 3. 性能优化
目的：提高算法在处理大文本时的效率。
实现：优化代码结构，减少不必要的计算，必要时使用多线程或异步处理（由于复杂性，示例中不做具体实现）。

# 二、详细实现
## 1. 使用 Nussinov 算法模拟 RNA 二级结构折叠

(1) Nussinov 算法简介
Nussinov 算法是一种动态规划算法，用于预测 RNA 序列的二级结构，通过最大化碱基配对数来计算最优结构。

(2) 在算法中的应用
输入：密码子序列（转换为碱基序列）。
输出：二级结构表示（配对矩阵或括号表示法）。
在加密中的作用：使用二级结构信息对序列进行重新排列或编码，增加非线性变换的复杂度。

(3) 代码实现
下面是 Nussinov 算法的简化实现：

```python
def nussinov_algorithm(sequence):
    n = len(sequence)
    dp = [[0]*n for _ in range(n)]
    traceback = [[0]*n for _ in range(n)]
    
    # 定义碱基配对规则
    def can_pair(b1, b2):
        pairs = [('A', 'U'), ('U', 'A'), ('C', 'G'), ('G', 'C'), ('G', 'U'), ('U', 'G')]
        return (b1, b2) in pairs
    
    # 动态规划填表
    for k in range(1, n):
        for i in range(n - k):
            j = i + k
            if can_pair(sequence[i], sequence[j]):
                pair = dp[i+1][j-1] + 1
            else:
                pair = dp[i+1][j-1]
            unpair = max(dp[i+1][j], dp[i][j-1])
            dp[i][j] = max(pair, unpair)
    
    # 生成二级结构（括号表示法）
    structure = ['.'] * n
    def traceback_structure(i, j):
        if i < j:
            if dp[i][j] == dp[i+1][j]:
                traceback_structure(i+1, j)
            elif dp[i][j] == dp[i][j-1]:
                traceback_structure(i, j-1)
            else:
                structure[i] = '('
                structure[j] = ')'
                traceback_structure(i+1, j-1)
    traceback_structure(0, n-1)
    return ''.join(structure)

```

(4) 在加密过程中的应用
在加密过程中，使用二级结构信息对序列进行重新排列。例如，根据配对信息交换碱基位置，或将配对的碱基组合编码。

## 2. 使用实际的生物序列作为密钥种子
(1) 获取实际生物序列
方法：用户提供生物序列，或从文件中读取。
示例：这里假设用户提供一段 RNA 序列作为密钥种子。
(2) 更新密钥生成函数

```python

def generate_dynamic_key_from_biological_data(seed_sequence):
    # 检查生物序列的有效性
    valid_bases = set('ACGU')
    if not set(seed_sequence.upper()).issubset(valid_bases):
        raise ValueError("Seed sequence contains invalid bases.")
    # 使用生物序列的哈希值作为密钥
    hash_object = hashlib.sha256(seed_sequence.encode())
    dynamic_key = hash_object.digest()  # 256位密钥
    return dynamic_key
```


# 3. 更新加密和解密函数
(1) 在加密函数中集成 Nussinov 算法
```python
#RNA二级结构折叠（使用Nussinov算法）
def apply_rna_secondary_structure(codon_sequence):
    # 将密码子序列展开为碱基序列
    base_sequence = ''.join(codon_sequence)
    # 使用Nussinov算法获取二级结构
    structure = nussinov_algorithm(base_sequence)
    # 根据结构重新排列序列（示例：将配对碱基放在一起）
    paired_indices = []
    unpaired_indices = []
    for i, s in enumerate(structure):
        if s == '(' or s == ')':
            paired_indices.append(i)
        else:
            unpaired_indices.append(i)
    # 重新排列序列
    new_sequence = ''.join([base_sequence[i] for i in paired_indices + unpaired_indices])
    # 将新的碱基序列转换回密码子序列
    new_codon_sequence = [new_sequence[i:i+3] for i in range(0, len(new_sequence), 3)]
    return new_codon_sequence
```

# (2) 在解密函数中逆转 Nussinov 算法的影响
由于 Nussinov 算法的非线性和复杂性，逆转过程较为困难。为了简化，我们在加密过程中记录重排的索引，以便在解密过程中恢复原始序列。

在加密函数中记录索引：

```python
def apply_rna_secondary_structure(codon_sequence):
    # ...（同上）
    # 记录新序列的索引顺序
    indices_order = paired_indices + unpaired_indices
    # 将新的碱基序列转换回密码子序列
    new_codon_sequence = [new_sequence[i:i+3] for i in range(0, len(new_sequence), 3)]
    return new_codon_sequence, indices_order
```

修改加密函数：

```python

def encrypt(plaintext, seed, seed_sequence):
    # ...（之前的步骤）
    # 步骤4
    folded_sequence, indices_order = apply_rna_secondary_structure(substituted_sequence)
    # ...（后续步骤）
    return encrypted_data_with_checksum, substitution_matrix, indices_order
在解密函数中使用索引恢复序列：
```

```python
def inverse_rna_secondary_structure(codon_sequence, indices_order):
    # 将密码子序列展开为碱基序列
    base_sequence = ''.join(codon_sequence)
    # 计算原序列长度
    original_length = len(base_sequence)
    # 创建一个列表用于存放恢复的碱基
    original_sequence = [''] * original_length
    # 恢复序列
    for i, idx in enumerate(indices_order):
        original_sequence[idx] = base_sequence[i]
    # 将碱基序列转换回密码子序列
    original_codon_sequence = [original_sequence[i:i+3] for i in range(0, original_length, 3)]
    return original_codon_sequence
```

```python
def decrypt(encrypted_data_with_checksum, seed, seed_sequence, substitution_matrix, indices_order):
    # ...（之前的步骤）
    # 步骤4
    unfolded_sequence = inverse_rna_secondary_structure(decrypted_sequence, indices_order)
    # ...（后续步骤）
    return plaintext
```



