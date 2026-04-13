# Imports
from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define a simple tool to get weather information for a city
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Create an agent with the specified model, tools, and system prompt
agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Invoke the agent with a user message asking about the weather in San Francisco
res = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

# Extract and print all messages from the agent's response
messages = res["messages"]

print("=== All messages ===")
for i, msg in enumerate(messages):
    print(f"\n[{i}] {type(msg).__name__}")
    print(msg.content)

print("\n=== Final answer ===")
print(messages[-1].content)