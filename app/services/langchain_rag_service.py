from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
import chromadb


class LangChainRAGService:
    """LangChain-based RAG service - DISABLED to prevent hanging"""
    
    def __init__(self):
        # Skip all initialization to prevent hanging
        self._vector_store = None
        self.embeddings = None
        self.text_splitter = None
    
    def _get_vector_store(self) -> Chroma:
        """Get or create the vector store - DISABLED to prevent hanging"""
        # Skip ChromaDB connection to prevent hanging - use in-memory fallback
        if self._vector_store is None:
            # Create a fallback in-memory store immediately without trying HTTP connection
            self._vector_store = Chroma(
                collection_name="interview_knowledge",
                embedding_function=self.embeddings
            )
        return self._vector_store
    
    def create_retriever(
        self,
        company: str,
        role: str,
        difficulty: str,
        question_type: str,
        k: int = 5
    ) -> BaseRetriever:
        """Create a targeted retriever for specific interview context"""
        
        vector_store = self._get_vector_store()
        
        # Create metadata filter for targeted retrieval
        filter_dict = {
            "company": company,
            "role": role,
            "difficulty": difficulty,
            "question_type": question_type
        }
        
        # Create retriever with search kwargs
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": 0.3,
                "filter": filter_dict
            }
        )
        
        return retriever
    
    async def get_interview_context(
        self,
        company: str,
        role: str,
        difficulty: str,
        question_type: str,
        transcript: str = ""
    ) -> Dict[str, Any]:
        """Get interview context - DISABLED to prevent hanging"""
        print(f"[DEBUG RAG] get_interview_context called - returning immediate fallback")
        # Return immediate fallback without any vector store operations
        return {
            "injected_context_text": f"Generate {difficulty} {question_type} questions for {role} position at {company}.",
            "source_documents": [],
            "source_info": [],
            "num_docs": 0
        }
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        company: str,
        role: str,
        difficulty: str,
        question_type: str
    ) -> List[str]:
        """Add new documents to the vector store"""
        
        vector_store = self._get_vector_store()
        
        # Convert to LangChain documents
        langchain_docs = []
        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue
                
            metadata = {
                "company": company,
                "role": role,
                "difficulty": difficulty,
                "question_type": question_type,
                "source": doc.get("source", "manual"),
                **doc.get("metadata", {})
            }
            
            langchain_docs.append(Document(page_content=content, metadata=metadata))
        
        # Split documents if needed
        if len(langchain_docs) > 0 and len(langchain_docs[0].page_content) > 1500:
            split_docs = self.text_splitter.split_documents(langchain_docs)
        else:
            split_docs = langchain_docs
        
        # Add to vector store
        ids = vector_store.add_documents(split_docs)
        return ids
    
    def create_hybrid_retriever(
        self,
        company: str,
        role: str,
        difficulty: str,
        question_type: str,
        transcript: str = ""
    ) -> BaseRetriever:
        """Create a hybrid retriever that combines semantic search with transcript analysis"""
        
        # Base semantic retriever
        semantic_retriever = self.create_retriever(company, role, difficulty, question_type)
        
        # If we have transcript, create a transcript retriever
        if transcript:
            transcript_docs = [
                Document(
                    page_content=transcript,
                    metadata={
                        "source": "transcript",
                        "company": company,
                        "role": role,
                        "type": "conversation_history"
                    }
                )
            ]
            
            # Create a temporary vector store for transcript
            from langchain_community.vectorstores import Chroma as TempChroma
            transcript_store = TempChroma.from_documents(
                documents=transcript_docs,
                embedding=self.embeddings,
                collection_name="temp_transcript"
            )
            
            transcript_retriever = transcript_store.as_retriever(
                search_kwargs={"k": 2}
            )
            
            # Create ensemble retriever (requires langchain-community)
            try:
                from langchain.retrievers import EnsembleRetriever
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[semantic_retriever, transcript_retriever],
                    weights=[0.7, 0.3]
                )
                return ensemble_retriever
            except ImportError:
                # Fallback to semantic retriever only
                pass
        
        return semantic_retriever


# Global instance
langchain_rag = LangChainRAGService()
