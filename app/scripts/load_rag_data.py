from app.services.langchain_rag_service import langchain_rag
from app.data.interview_knowledge import INTERVIEW_DOCS


def load_data():
    print("🔄 FULL RESET of RAG data...")

    # 🔥 Step 1: TRUE FULL DELETE
    langchain_rag.reset_database()   # ✅ use full wipe

    # 🔥 Step 2: Add fresh data
    for doc in INTERVIEW_DOCS:
        print(f"Adding document: {doc}")

        langchain_rag.add_documents(
            documents=[doc],
            company=doc.get("company", "General"),
            role=doc.get("role", "General"),
            difficulty=doc.get("difficulty", "Medium"),
            question_type=doc.get("type", "conceptual")
        )

    print("✅ RAG data fully refreshed!")


if __name__ == "__main__":
    load_data()