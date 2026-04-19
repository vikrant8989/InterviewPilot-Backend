from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LangChainRAGService:
    """Production-ready RAG using local HuggingFace embeddings (no API key required)"""

    def __init__(self):
        # ✅ Local embeddings (works with Groq setup)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ✅ Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        # ✅ Persistent vector store
        self._vector_store = Chroma(
            collection_name="interview_knowledge",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )

    def create_retriever(
        self,
        company: str,
        role: str,
        difficulty: str,
        question_type: str,
        k: int = 5
    ) -> BaseRetriever:
        """Create filtered retriever"""
        filter_dict = {
            "$and": [
                {"$or": [{"company": company}, {"company": "General"}]},
                {"$or": [{"role": role}, {"role": "General"}]},
                {"$or": [{"question_type": question_type}, {"question_type": "conceptual"}]}
            ]
        }

        return self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "filter": filter_dict
            }
        )

    async def get_interview_context(
        self,
        company: str,
        role: str,
        difficulty: str,
        question_type: str,
        transcript: str = ""
    ) -> Dict[str, Any]:
        """Retrieve real RAG context"""

        try:
            retriever = self.create_retriever(
                company, role, difficulty, question_type
            )

            # Use transcript if available else fallback query
            query = transcript if transcript else f"{company} {role} {question_type}"

            docs = retriever.invoke(query)

            if not docs:
                return {
                    "injected_context_text": "",
                    "source_documents": [],
                    "num_docs": 0
                }

            context = "\n".join([doc.page_content for doc in docs])

            return {
                "injected_context_text": context,
                "source_documents": docs,
                "num_docs": len(docs)
            }

        except Exception as e:
            print(f"[RAG ERROR]: {e}")
            return {
                "injected_context_text": "",
                "source_documents": [],
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
        """Add documents to vector DB"""

        langchain_docs = []

        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue
                
            doc_question_type = doc.get("question_type") or doc.get("type", "conceptual")
            doc_difficulty = doc.get("difficulty", difficulty)

            metadata = {
                "company": doc.get("company", company),
                "role": doc.get("role", role),
                "difficulty": doc_difficulty,
                "question_type": doc_question_type,
                "source": doc.get("source", "manual"),
                "tags": doc.get("tags", [])
            }
            langchain_docs.append(Document(
                page_content=content,
                metadata=metadata
            ))

        # Split large docs
        split_docs = self.text_splitter.split_documents(langchain_docs)

        ids = self._vector_store.add_documents(split_docs)

        return ids
    def reset_database(self):
        """Full wipe of vector DB"""
        try:
            self._vector_store._collection.delete(where={})
            print("🔥 FULL DB RESET DONE")
        except Exception as e:
            print(f"[RESET ERROR]: {e}")
            
    def update_documents(
        self,
        documents: List[Dict[str, Any]],
        company: str,
        role: str,
        difficulty: str,
        question_type: str
    ):
        """Scoped update (delete old + add new)"""
        try:
            # Step 1: Delete old documents (scoped)
            self.delete_documents(
                company=company,
                role=role,
                question_type=question_type
            )

            # Step 2: Add new documents
            return self.add_documents(
                documents,
                company,
                role,
                difficulty,
                question_type
            )

        except Exception as e:
            print(f"[UPDATE ERROR]: {e}")
            return []

# Global instance
langchain_rag = LangChainRAGService()