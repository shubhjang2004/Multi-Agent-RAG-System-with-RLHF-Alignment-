#  Multi-Agent Orchestration



from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from pathlib import Path
import json

CHROMA_DIR = Path("data/vector_store")

print("Multi-agent setup initialized")


# Load vector store
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
    collection_name="arxiv_papers"
)

print("Vector store loaded")


# Retrieval Agent Tool
def retrieval_tool(query: str) -> str:
    """Search research papers for relevant information"""
    docs = vectorstore.similarity_search(query, k=5)
    
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"{i}. {doc.metadata['title']}\n{doc.page_content[:200]}...\n")
    
    return "\n".join(results)

retrieval_agent_tool = Tool(
    name="PaperSearch",
    func=retrieval_tool,
    description="Search academic papers for information about a topic"
)


# Citation Agent Tool
def citation_tool(arxiv_ids: str) -> str:
    """Format citations for papers"""
    ids = arxiv_ids.split(',')
    citations = []
    
    for arxiv_id in ids:
        # Query metadata
        results = vectorstore.get(
            where={"arxiv_id": arxiv_id.strip()}
        )
        if results and results['metadatas']:
            meta = results['metadatas'][0]
            citation = f"[{meta['arxiv_id']}] {meta['title']} - {meta['authors']}"
            citations.append(citation)
    
    return "\n".join(citations)

citation_agent_tool = Tool(
    name="FormatCitations",
    func=citation_tool,
    description="Format academic citations for papers given arxiv IDs"
)


# Test tools
print("Testing retrieval tool...")
result = retrieval_tool("transformers in NLP")
print(result[:300])

print("\n" + "="*60)
print("Multi-agent orchestration ready!")
print("Notebook 05 Complete!")