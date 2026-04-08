# Output Parsers: Structuring LLM responses
# Output parsers transform the raw LLM text output into structured formats
# like lists, JSON, or Pydantic models.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- StrOutputParser: extracts string content from AIMessage ---
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}.")
chain = prompt | llm | StrOutputParser()

# Without parser: returns AIMessage object
# With parser: returns plain string
result = chain.invoke({"topic": "programming"})
print("=== StrOutputParser (returns plain string) ===")
print(result)
print(f"Type: {type(result)}")

# --- JsonOutputParser: parse response as JSON ---
class Movie(BaseModel):
    title: str = Field(description="The movie title")
    year: int = Field(description="The release year")
    genre: str = Field(description="The movie genre")

json_parser = JsonOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_template(
    "Suggest a classic movie.\n{format_instructions}"
)
chain = prompt | llm | json_parser

result = chain.invoke({"format_instructions": json_parser.get_format_instructions()})
print("\n=== JsonOutputParser (returns dict) ===")
print(result)
print(f"Type: {type(result)}")

# --- Chaining: Prompt | Model | Parser (complete LCEL pipeline) ---
prompt = ChatPromptTemplate.from_template(
    "List 3 benefits of {topic}. Return as JSON array of strings."
)
chain = prompt | llm | JsonOutputParser()
result = chain.invoke({"topic": "exercise"})
print("\n=== JSON array output ===")
print(result)
