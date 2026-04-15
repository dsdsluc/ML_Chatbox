from flask import Flask, render_template, request
from src.helper import get_embedding
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from src.prompt import *
import os
from store import docs

app = Flask(__name__)

load_dotenv()

# API KEY
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Embedding (không cần dùng lại ở đây nhưng giữ cũng ok)
embeddings = get_embedding()

# Retriever
retriever = docs.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

# Prompt
base_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# ✅ RAG chain (NEW)
rag_chain = (
    {
        "context": retriever,
        "input": RunnablePassthrough()
    }
    | base_prompt
    | llm
)

# Routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]

    response = rag_chain.invoke(msg)

    print("Response: ", response.content)

    return str(response.content)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)