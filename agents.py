# Agents: LLMs that can reason and take actions
# Agents use LLMs to decide WHICH tools to call and in WHAT ORDER.
# Unlike chains (fixed sequence), agents dynamically choose their actions.

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

# --- Define tools the agent can use ---
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "london": "15°C, Cloudy",
        "tokyo": "22°C, Sunny",
        "new york": "18°C, Partly Cloudy",
        "san francisco": "16°C, Foggy",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Example: '2 + 3 * 4'"""
    allowed = set("0123456789+-*/(). ")
    if all(c in allowed for c in expression):
        return str(eval(expression))
    return "Invalid expression"

@tool
def search_knowledge(query: str) -> str:
    """Search a knowledge base for information."""
    knowledge = {
        "langchain": "LangChain is a framework for building LLM applications.",
        "python": "Python is a high-level programming language created by Guido van Rossum.",
        "react": "ReAct is a prompting pattern where the agent Reasons and Acts iteratively.",
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return "No relevant information found."

# --- Create an agent using LangChain ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_agent(
    model=llm,
    tools=[get_weather, calculate, search_knowledge],
)

# --- Ask the agent questions (it decides which tools to use) ---
print("=== Agent: Weather question ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]}
)
for msg in result["messages"]:
    print(f"[{type(msg).__name__}] {msg.content}")

print("\n=== Agent: Math question ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is 42 * 58?"}]}
)
for msg in result["messages"]:
    print(f"[{type(msg).__name__}] {msg.content}")

print("\n=== Agent: Knowledge question ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is LangChain?"}]}
)
for msg in result["messages"]:
    print(f"[{type(msg).__name__}] {msg.content}")

print("\n=== Agent: Multi-step question ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in London and what is 100 / 4?"}]}
)
print("\nFinal answer:", result["messages"][-1].content)
