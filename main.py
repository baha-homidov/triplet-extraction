# Assume openai>=1.0.0
from openai import OpenAI


# Create an OpenAI client with your deepinfra token and endpoint
from dotenv import load_dotenv
import os

load_dotenv()

openai = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai",
)

chat_completion = openai.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
)

print(chat_completion.choices[0].message.content)
print(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)


