# Tool Calling: Letting LLMs decide which tools to use
# Tool calling (function calling) allows the LLM to choose which tool to call
# and with what arguments, based on the user's question.

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# --- Define tools ---
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "london": "15°C, Cloudy",
        "tokyo": "22°C, Sunny",
        "new york": "18°C, Partly Cloudy",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '2 + 3 * 4'"""
    allowed = set("0123456789+-*/(). ")
    if all(c in allowed for c in expression):
        return str(eval(expression))
    return "Invalid expression"

# --- Bind tools to the model ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([get_weather, calculate])

# --- LLM decides to call a tool ---
response = llm_with_tools.invoke("What's the weather in Tokyo?")
print("=== LLM response with tool call ===")
print(f"Content: {response.content}")
print(f"Tool calls: {response.tool_calls}")

# --- Execute the tool call ---
if response.tool_calls:
    tool_call = response.tool_calls[0]
    print(f"\n=== Executing tool: {tool_call['name']} ===")
    print(f"Args: {tool_call['args']}")

    # Map tool names to tool functions
    tool_map = {"get_weather": get_weather, "calculate": calculate}
    selected_tool = tool_map[tool_call["name"]]
    tool_result = selected_tool.invoke(tool_call["args"])
    print(f"Result: {tool_result}")

# --- LLM decides to call a different tool ---
response2 = llm_with_tools.invoke("What is 15 * 7 + 23?")
print(f"\n=== Math question tool call ===")
print(f"Tool calls: {response2.tool_calls}")
if response2.tool_calls:
    tool_call = response2.tool_calls[0]
    tool_map = {"get_weather": get_weather, "calculate": calculate}
    result = tool_map[tool_call["name"]].invoke(tool_call["args"])
    print(f"Result: {result}")

# --- No tool needed: LLM answers directly ---
response3 = llm_with_tools.invoke("Hello, how are you?")
print(f"\n=== No tool needed ===")
print(f"Content: {response3.content}")
print(f"Tool calls: {response3.tool_calls}")
