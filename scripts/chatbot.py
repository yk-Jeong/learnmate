from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 OPENAI_API_KEY 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# OpenAIEmbeddings 생성 시 API 키 전달
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

# PDF 파일 로드하여 텍스트 추출 (파일 경로를 지정)
def load_pdf(file_path):
    try:
        loader = UnstructuredPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"PDF 로딩 중 오류 발생: {str(e)}")
        return []
        
# PDF 파일 로드
documents = load_pdf("/Users/jeong/AI/learnmate/data/papers/test_2.pdf")

if not documents or all(len(d.page_content.strip()) == 0 for d in documents):
    raise ValueError("PDF에 텍스트가 없거나 로딩에 실패했습니다.")

# 디버깅 출력
print(f"{len(documents)}개의 문서를 로드했습니다.")
if documents:
    print(f"첫 문서 일부:\n{documents[0].page_content[:300]}")

# 임베딩 생성
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

# FAISS 인덱스 생성 및 저장
db = FAISS.from_documents(documents, embedding)  # 벡터 DB 생성
db.save_local("vectordb")  # 벡터 DB 저장

# FAISS 인덱스 로드
try:
    db = FAISS.load_local("vectordb", embedding, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
except Exception as e:
    raise ValueError(f"FAISS 로딩 오류: {str(e)}")

# LLM 정의
llm = ChatOpenAI(
    temperature=0.2,
    model_name="gpt-3.5-turbo",
    openai_api_key=openai_api_key
)

# QA 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 기본값이긴 하나 명시해도 좋음
    retriever=retriever,
    return_source_documents=True,
    input_key="query"
)

# 질문 처리 함수
def ask_rag(question):
    try:
        result = qa_chain({"query": question})
        return result["result"]
    except Exception as e:
        print(f"오류 발생: {str(e)}")  # 에러 메시지 출력
        return f"오류가 발생했습니다: {str(e)}"

