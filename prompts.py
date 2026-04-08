# Prompts: Prompt Templates in LangChain
# Prompt templates help create reusable, parameterized prompts
# instead of hardcoding strings every time.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Simple PromptTemplate (for plain text) ---
simple_prompt = PromptTemplate.from_template(
    "Tell me a fun fact about {topic}."
)
formatted = simple_prompt.format(topic="elephants")
print("=== Formatted simple prompt ===")
print(formatted)

# --- ChatPromptTemplate (for chat models) ---
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {field}. Answer in {max_words} words or less."),
    ("human", "{question}"),
])

# Format and invoke
messages = chat_prompt.format_messages(
    field="astronomy",
    max_words="20",
    question="Why do stars twinkle?"
)
response = llm.invoke(messages)
print("\n=== Chat prompt response ===")
print(response.content)

# --- Prompt + Model chain using pipe operator (LCEL) ---
chain = chat_prompt | llm
response = chain.invoke({
    "field": "biology",
    "max_words": "15",
    "question": "What is DNA?"
})
print("\n=== Prompt | Model chain response ===")
print(response.content)
