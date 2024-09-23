# 3.0改进
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

