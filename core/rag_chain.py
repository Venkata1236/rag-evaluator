# core/rag_chain.py
# RAG pipeline to evaluate
# Concept: Build FAISS RAG chain — same as before but now we'll evaluate it

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def build_rag_chain(api_key: str, document_path: str = "data/hr_policies.txt"):
    """
    Builds a RAG chain from a text document.

    Args:
        api_key      : OpenAI API key
        document_path: path to the document

    Returns:
        tuple of (chain, retriever)
    """
    # Load document
    with open(document_path, "r") as f:
        text = f.read()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.create_documents([text])
    print(f"✅ Split document into {len(chunks)} chunks")

    # Create embeddings and FAISS vector store
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=api_key
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("✅ FAISS vector store built")

    # Prompt
    prompt = PromptTemplate.from_template("""
You are an HR assistant. Answer questions based ONLY on the provided context.
If the answer is not in the context, say "I couldn't find that information."

Context:
{context}

Question: {question}

Answer:
""")

    # LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.0,
        openai_api_key=api_key
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # LCEL chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever