from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


pdf_dir = "data/papers"
vector_store_dir = "vectordb"

all_docs = []

for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_dir, filename))
        docs = loader.load()
        all_docs.extend(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = text_splitter.split_documents(all_docs)

db = FAISS.from_documents(split_docs, embedding)
db.save_local(vector_store_dir)

print("✅ 벡터 저장 완료!")
