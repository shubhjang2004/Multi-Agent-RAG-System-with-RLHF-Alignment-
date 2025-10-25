#  Basic RAG Pipeline


import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pathlib import Path

CHROMA_DIR = Path("data/vector_store")

print("RAG pipeline setup")


# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}
)

# Load vector store
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
    collection_name="arxiv_papers"
)

print(f"Vector store loaded: {vectorstore._collection.count()} documents")


# Create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={
        'k': 5,
        'fetch_k': 20,
        'lambda_mult': 0.7
    }
)

print("Retriever configured")

# Test retrieval
test_queries = [
    "What are attention mechanisms in neural networks?",
    "How does BERT work?",
    "Explain reinforcement learning from human feedback",
    "What is the transformer architecture?"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    docs = retriever.get_relevant_documents(query)
    print(f"Retrieved {len(docs)} documents")
    print(f"Top result: {docs[0].metadata['title']}")

