# test_batched.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sliding_window_ensemble import SWEAWithCache

# モデル読み込み（軽めのモデルでテスト）
model_name = "Qwen/Qwen2.5-1.5B-Instruct"   # 軽くて速いモデル
# model_name = "meta-llama/Llama-3.2-1B-Instruct"  # こちらでもOK

print("モデルを読み込んでいます...（初回は少し時間がかかります）")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# SWEAの初期化（バッチ化版）
swea = SWEAWithCache(
    model=model,
    tokenizer=tokenizer,
    sink_size=10,
    window_size=512,
    local_size=100,
    ensemble_n=1.5
)

# テスト用プロンプト
prompt = "日本の美しい四季について、詳しく説明してください。"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

print("\n=== 生成開始 ===")
print(f"入力長: {input_ids.shape[1]} トークン")

# 10トークン生成してテスト
generated = []
for i in range(10):
    logits, mode = swea.predict_next_logits(input_ids)
    
    next_token = torch.argmax(logits, dim=-1)
    generated.append(next_token.item())
    
    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    print(f"Step {i+1:2d} | Mode: {mode} | Token: {tokenizer.decode(next_token)}")

print("\n生成結果:", tokenizer.decode(generated))
print("テスト完了！")
