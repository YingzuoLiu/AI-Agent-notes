# 端到端 Demo: MCP + Prefix + GPT-2 推理

!pip install transformers torch pandas -q

import torch
import torch.nn as nn
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============================
# 1. MCP 模拟层：读取表格 & 生成上下文向量
# ============================
class SimpleMCP:
    def __init__(self, hidden_size):
        # 一个简单的表格encoder，用平均+MLP编码
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_size),  # 输入是一列数字
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def get_context_vector(self, csv_path, column):
        df = pd.read_csv(csv_path)
        print("表格内容:\n", df)
        values = df[column].values.reshape(-1, 1)
        values_tensor = torch.tensor(values, dtype=torch.float32)
        print("取出列:", column, "张量形状:", values_tensor.shape)
        mean_val = values_tensor.mean(dim=0)  # 求平均
        context_vector = self.encoder(mean_val)
        print("MCP 输出向量 shape:", context_vector.shape)
        return context_vector

# 造一个简单的表格 (假设我们有用户年龄)
df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [25, 32, 29]})
csv_path = "/mnt/data/users.csv"
df.to_csv(csv_path, index=False)

# 初始化MCP，获取表格向量
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
hidden_size = model.config.hidden_size

mcp = SimpleMCP(hidden_size)
table_vector = mcp.get_context_vector(csv_path, "age")  # 取age列
prefix = table_vector.unsqueeze(0).unsqueeze(1)  # [batch=1, seq=1, hidden]
print("prefix shape:", prefix.shape)

# ============================
# 2. Prefix 注入层
# ============================
prompt = "Who is the oldest person in the table?"
prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
prompt_embeds = model.transformer.wte(prompt_ids)
print("prompt_embeds shape:", prompt_embeds.shape)

# 拼接 prefix + prompt embedding
input_embeds = torch.cat([prefix, prompt_embeds], dim=1)
print("拼接后 input_embeds shape:", input_embeds.shape)

# ============================
# 3. 推理层：生成文本
# ============================
output_ids = model.generate(inputs_embeds=input_embeds, max_length=40)
print("模型输出:", tokenizer.decode(output_ids[0], skip_special_tokens=True))
