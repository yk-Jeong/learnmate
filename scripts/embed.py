import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

MD_DIR = "/Users/jeong/AI/learnmate/data/abstracts"
VECTOR_DIR = "vectordb_md"

def load_md_documents(md_dir):
    documents = []
    for filename in os.listdir(md_dir):
        if filename.endswith(".md"):
            filepath = os.path.join(md_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))
    return documents

def build_vector_db(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedding)
    db.save_local(VECTOR_DIR)
    print(f"✅ 벡터 DB 저장 완료: {VECTOR_DIR} (총 {len(chunks)}개 조각)")

if __name__ == "__main__":
    print("📂 MD 요약 파일 로딩 중...")
    docs = load_md_documents(MD_DIR)
    if not docs:
        print("❌ .md 문서가 없습니다.")
    else:
        print(f"📄 {len(docs)}개 문서 로딩 완료")
        build_vector_db(docs)
