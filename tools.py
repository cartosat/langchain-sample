# Tools: Defining custom tools for LLMs
# Tools give LLMs the ability to interact with the outside world -
# call APIs, do calculations, search databases, etc.

from langchain_core.tools import tool, StructuredTool
from dotenv import load_dotenv

load_dotenv()

# --- Define tools using @tool decorator (simplest way) ---
@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@tool
def get_word_count(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split())

# --- Inspect tool properties ---
print("=== Tool properties ===")
print(f"Name: {add.name}")
print(f"Description: {add.description}")
print(f"Args schema: {add.args}")

# --- Invoke tools directly ---
print("\n=== Direct tool invocation ===")
print(f"add(3, 5) = {add.invoke({'a': 3, 'b': 5})}")
print(f"multiply(4, 7) = {multiply.invoke({'a': 4, 'b': 7})}")
print(f"word_count('hello world foo') = {get_word_count.invoke({'text': 'hello world foo'})}")

# --- StructuredTool: create tool from a function programmatically ---
def power(base: int, exponent: int) -> int:
    """Raise base to the power of exponent."""
    return base ** exponent

power_tool = StructuredTool.from_function(
    func=power,
    name="power",
    description="Raise base to the power of exponent",
)
print(f"\n=== StructuredTool ===")
print(f"power(2, 10) = {power_tool.invoke({'base': 2, 'exponent': 10})}")

# --- List all tools ---
all_tools = [add, multiply, get_word_count, power_tool]
print("\n=== All available tools ===")
for t in all_tools:
    print(f"  - {t.name}: {t.description}")
