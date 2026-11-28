import os
import httpx
import asyncio
import tiktoken
from enum import Enum
from pydantic import BaseModel
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic


class Provider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"
    ollama = "ollama"

class InferenceRequest(BaseModel):
    provider: Provider
    model: str
    messages: list[dict]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 512
    
class InferenceResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    

class BaseClient:
    async def generate(self, req: InferenceRequest)-> InferenceResponse:
        ...
    
    
class OllamaClient(BaseClient):
    def __init__(self, base_url="http://localhost:11434"):
        self.http =  httpx.AsyncClient(base_url=base_url, timeout=60)
        
    async def generate(self, req: InferenceRequest):
        payload = {
            "model": req.model or "llama3.2",
            "messages": req.messages,
            "options": {
                "temperature": req.temperature,
                "top_p": req.top_p
            },
            "stream": False,
        }
        r = await self.http.post("/api/chat", json=payload)
        r.raise_for_status()
        d = r.json()
        return InferenceResponse(
            text=d["message"]["content"], 
            input_tokens=d.get("prompt_eval_count", 0),
            output_tokens=d.get("eval_count", 0)
        )
        

class OpenAIClient(BaseClient):
    def __init__(self, api_key=None):
        self._client = None
        self._model = "gpt-4o-mini"
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._system_prompt = "You're a conversational agent"
    
    async def generate(self, req: InferenceRequest):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self._client = AsyncOpenAI(api_key=self._api_key)
            
        resp = self._client.responses.create(
            instructions=self._system_prompt,
            input=trim_context(req.messages, 120000),
            temperature=req.temperature,
            max_output_tokens=req.max_tokens,
        )
        out = [{"role": "assistant", "content": resp.output_text}]
        return InferenceResponse(
            text=resp.output_text,
            input_tokens=count_tokens(req.messages, model=self._model),
            output_tokens=count_tokens(out, model=self._model)
        )
       
class AnthropicClient(BaseClient):
    def __init__(self):
        self._model = None
        self._api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._client = AsyncAnthropic(api_key=self._api_key)
    
    async def generate(self):
        return
    
    
def count_tokens(messages: list[dict], model="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for m in messages:
        total += 4
        total += len(enc.encode(m.get("content", "")))
        total += len(enc.encode(m.get("role", "")))
    return total + 2
    
def trim_context(messages, max_tokens, model="gpt-4o-mini"):
    system = [m for m in messages if m["role"] == "system"]
    others = [m for m in messages if m["role"] != "system"]
    while others and count_tokens(system + others, model) > max_tokens:
        others.pop(0)
    return system + others


clients = {
    Provider.ollama: OllamaClient(), 
    Provider.openai: OpenAIClient(),
    Provider.anthropic: AnthropicClient()
}

async def run(req):
    return await clients[req.provider].generate(req)


messages = [
    {"role": "assistant", "content": "Call me Jarvis, your faithful AI assistant."},
    {"role": "user", "content": "What is the best way to become a robotics AI Engineer?"}
]


if __name__ == "__main__":
    req = InferenceRequest(
        provider="ollama",
        model="llama3.2",
        messages=messages,
        temperature=0.7,
        top_p=1.0,
        max_tokens=512
    )
    resp = asyncio.run(run(req))
    print(resp)