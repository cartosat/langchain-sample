# Runnables: LangChain Expression Language (LCEL)
# Runnables are the building blocks of LCEL - any component that implements
# invoke(), stream(), batch() is a Runnable. They can be chained with the | operator.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Basic chain with | (pipe) operator ---
prompt = ChatPromptTemplate.from_template("Explain {concept} simply in one sentence.")
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"concept": "recursion"})
print("=== Basic chain ===")
print(result)

# --- RunnableLambda: wrap any function as a Runnable ---
def word_count(text: str) -> str:
    return f"{text}\n\n[Word count: {len(text.split())}]"

chain_with_count = prompt | llm | StrOutputParser() | RunnableLambda(word_count)
result = chain_with_count.invoke({"concept": "blockchain"})
print("\n=== RunnableLambda (adds word count) ===")
print(result)

# --- RunnablePassthrough: pass input through unchanged ---
# Useful when you want to forward original input alongside transformed data
chain = RunnablePassthrough() | RunnableLambda(lambda x: f"You said: {x}")
result = chain.invoke("hello")
print("\n=== RunnablePassthrough ===")
print(result)

# --- RunnableParallel: run multiple chains simultaneously ---
joke_prompt = ChatPromptTemplate.from_template("Tell a one-line joke about {topic}.")
fact_prompt = ChatPromptTemplate.from_template("Tell one fun fact about {topic}.")

parallel_chain = RunnableParallel(
    joke=joke_prompt | llm | StrOutputParser(),
    fact=fact_prompt | llm | StrOutputParser(),
)
result = parallel_chain.invoke({"topic": "cats"})
print("\n=== RunnableParallel (joke + fact simultaneously) ===")
print(f"Joke: {result['joke']}")
print(f"Fact: {result['fact']}")

# --- Streaming: process response tokens as they arrive ---
print("\n=== Streaming ===")
stream_chain = prompt | llm | StrOutputParser()
for chunk in stream_chain.stream({"concept": "machine learning"}):
    print(chunk, end="", flush=True)
print()
