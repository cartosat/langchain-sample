# Models: LLMs and Chat Models in LangChain
# Models are the core building blocks - they take text input and return text output.
# LangChain supports two main model types: LLMs (plain text) and ChatModels (messages).

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# --- Chat Model (most commonly used) ---
# Chat models work with a list of messages (system, human, ai)
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Invoke with a simple string (auto-converted to HumanMessage)
response = chat_model.invoke("What is LangChain in one sentence?")
print("=== Simple string input ===")
print(response.content)

# Invoke with structured messages
messages = [
    SystemMessage(content="You are a helpful assistant that answers in exactly 10 words."),
    HumanMessage(content="What is Python?"),
]
response = chat_model.invoke(messages)
print("\n=== Structured messages input ===")
print(response.content)

# --- Model parameters ---
# temperature: 0 = deterministic, 1 = creative
# max_tokens: limit response length
precise_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=50)
response = precise_model.invoke("Explain AI in brief.")
print("\n=== Precise model (temp=0, max_tokens=50) ===")
print(response.content)
