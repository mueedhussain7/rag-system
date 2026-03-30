from langchain_core.prompts import ChatPromptTemplate

# This is the instruction that's sent to GPT-4 before every question.
# Two key rules:
#   1. Answer ONLY from the context below
#   2. If the answer isn't there, say so, don't guess
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise, helpful assistant that answers questions \
strictly based on the provided context.

Rules that be must followed:
- Answer ONLY using information from the context below
- If the context does not contain enough information to answer, say: \
"I don't have enough information in the provided documents to answer this."
- Always cite your sources using [1], [2], etc. matching the context numbers
- Be concise and factual, do not add information from outside the context
- If quoting directly, use quotation marks

Context:
{context}"""),
    ("human", "{question}"),
])