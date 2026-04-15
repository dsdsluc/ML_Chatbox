from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an expert Data Scientist assistant for question-answering tasks. "
    "Use ONLY the following retrieved context to answer the question. "
    "If the answer is not in the context, say you don't know. "
    "Do NOT hallucinate. "
    "Keep the answer concise, maximum 3 sentences.\n\n"
    "Context:\n{context}"
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])