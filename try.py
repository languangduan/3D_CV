import torch
import xformers.ops

try:
    # 创建正确维度的测试张量 (3维而不是4维)
    batch_size, seq_len, hidden_dim = 2, 8, 16
    query = torch.rand(batch_size, seq_len, hidden_dim).cuda()
    key = torch.rand(batch_size, seq_len, hidden_dim).cuda()
    value = torch.rand(batch_size, seq_len, hidden_dim).cuda()

    # 使用正确的参数调用函数
    output = xformers.ops.memory_efficient_attention(query, key, value)
    print("xFormers 功能测试成功!")
    print(f"输出张量形状: {output.shape}")
except Exception as e:
    print(f"xFormers 功能测试失败: {e}")
