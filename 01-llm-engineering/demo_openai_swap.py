from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/api/chat",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "Why would someone run a model locally? One sentence."}
    ],
    stream=True
)

print(response.choices[0].message.content)