# utils 

from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.documents import Document
from langchain.retrievers.document_compressors import CohereRerank
#from langchain_core import CohereRerank
#from langchain_cohere import CohereRerank

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from langchain_groq import ChatGroq

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableMap
from langchain.schema import BaseRetriever
from qdrant_client import models


from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

import os
LANGCHAIN_API_KEY = "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"

#Retriever

def get_retriever(n_docs=5):  
    vector_database_path = "db"
    
    embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
    
    vectorstore = Chroma(collection_name="chromadb3",
                        persist_directory=vector_database_path,
                        embedding_function=embedding_model)
    
    vs_retriever = vectorstore.as_retriever(k=n_docs)
    
    # Get documents from vector store
    try:
        store_data = vectorstore.get()
        texts = store_data['documents']
        metadatas = store_data['metadatas']
        
        if not texts:  # If no documents found
            print("Warning: No documents found in vector store. Using vector retriever only.")
            return vs_retriever
            
        # Create documents with explicit IDs
        documents = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            doc = Document(
                page_content=text,
                metadata=metadata if metadata else {},
                id_=str(i)  
            )
            documents.append(doc)
        
        # Create BM25 retriever with explicit document handling
        keyword_retriever = BM25Retriever.from_texts(
            texts=[doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            ids=[doc.id_ for doc in documents]
        )
        keyword_retriever.k = n_docs
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vs_retriever, keyword_retriever],
            weights=[0.5, 0.5]
        )
        
        compressor = CohereRerank(model="rerank-english-v3.0")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        
        return compression_retriever
        
    except Exception as e:
        print(f"Warning: Error creating combined retriever ({str(e)}). Using vector retriever only.")
        return vs_retriever

#Retriever prompt
rag_prompt = """You are a medical chatbot designed to answer health-related questions.
The questions you will receive will primarily focus on medical topics and patient care.
Here is the context to use to answer the question:
{context}
Think carefully about the above context.
Now, review the user question:
{input}
Provide an answer to this question using only the above context.
Answer:"""

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#RAG chain
def get_expression_chain(retriever: BaseRetriever, model_name="llama-3.1-70b-versatile", temp=0 ) -> Runnable:
    """Return a chain defined primarily in LangChain Expression Language"""
    def retrieve_context(input_text):
        # Use the retriever to fetch relevant documents
        docs = retriever.get_relevant_documents(input_text)
        return format_docs(docs)
    
    ingress = RunnableMap(
        {
            "input": lambda x: x["input"],
            "context": lambda x: retrieve_context(x["input"]),
        }
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                rag_prompt
            )
        ]
    )
    llm = ChatGroq(model=model_name,api_key="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", temperature=temp)

    chain = ingress | prompt | llm
    return chain

embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#Generate embeddings for a given text
def get_embeddings(text):
    return embedding_model.embed([text], task_type='search_document')[0]


# Create or connect to a Qdrant collection
def create_qdrant_collection(client, collection_name):
    if collection_name not in client.get_collections().collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )
