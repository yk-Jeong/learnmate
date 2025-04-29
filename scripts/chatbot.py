import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
import argparse

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

pdf_dir = "/Users/jeong/AI/learnmate/data/abstracts"
file_count = len([f for f in os.listdir(pdf_dir) if os.path.isfile(os.path.join(pdf_dir, f))])


# 환경 설정
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# 벡터 DB 로드
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
try:
    db = FAISS.load_local("vectordb", embedding, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 2})
except Exception as e:
    raise ValueError(f"❌ FAISS 로딩 오류: {e}")

# LLM 설정
llm = ChatOpenAI(
    temperature=0.2,
    model_name="gpt-4.1", 
    openai_api_key=openai_api_key
)

# 시스템 프롬프트 설정
system_prompt = (
    "your persona: 중고등학생의 학업 고민을 들어주는 따뜻하고 전략적인 대학생 멘토"
    "아래 문서를 참고해서, 선배가 후배에게 조언하듯 친절하게 제안"
    "먼저 질문자의 상황에 공감하고, 이어서 [전략 → 실천] 구조로 답변"
    "텍스트+멀티모달 출력(예시카드 등등)"
    "해요체로 작성"
    "/Users/jeong/AI/learnmate/data/abstracts 내의 데이터에 기반할 것"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "질문: {question}\n\n📎 참고 문서:\n{context}")
])

# QA 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# 질문 처리 함수
def ask_rag(question):
    try:
        result = qa_chain.invoke({"query": question})
        return result["result"]
    except Exception as e:
        print(f"❌ 오류: {e}")
        return f"오류가 발생했습니다: {e}"

# Streamlit UI
st.set_page_config(page_title="Learnmate", layout="wide")
st.title("Learnmate: 학습 고민을 나누어요!")
st.markdown(f"{file_count}편의 실제 논문에 기반해서, 당신의 학습 고민을 과학적으로 해소해 드립니다.")

question = st.text_input("무엇을 도와 드릴까요? 구체적으로 질문할수록, 자세하게 대답해 드릴 수 있어요!")

if question:
    with st.spinner("대답을 고민하는 중..."):
        answer = ask_rag(question)
        st.success("이렇게 해 봐요!")
        st.write(answer)
