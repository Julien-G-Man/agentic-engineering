from llama_cpp import Llama

MODEL_PATH = "C:\Users\user\.ollama\models\Qwen3.5-9B-Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    verbose=False
)

response = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Write a haikyu about swimming in the stars"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response["choices"][0]["message"]["content"])
