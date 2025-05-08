import os
import time
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from sklearn.metrics.pairwise import cosine_similarity

# === 환경 설정 === #
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 병렬 경고 제거
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
embedding_model_name = os.getenv("EMBEDDING_MODEL")

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")
if not embedding_model_name:
    raise ValueError("❌ EMBEDDING_MODEL not found in .env file")

# === 임베딩 및 DB 로딩 === #
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

db = FAISS.load_local("vectordb", embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

# === LLM 설정 === #
llm = ChatOpenAI(
    temperature=0.2,
    model_name="gpt-4.1",
    openai_api_key=openai_api_key
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# === 평가 함수 === #
def cosine_sim(text1, text2, embedding_model):
    emb1 = embedding_model.embed_documents([text1])[0]
    emb2 = embedding_model.embed_documents([text2])[0]
    return cosine_similarity([emb1], [emb2])[0][0]

def evaluate_accuracy(retrieved, expected, embedding_model, threshold=0.4):
    tp, fn = 0, 0
    for e in expected:
        if any(cosine_sim(r, e, embedding_model) >= threshold for r in retrieved):
            tp += 1
        else:
            fn += 1
    fp = len(retrieved) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return prec, rec, f1

def evaluate_query(query, expected, embedding_model, retriever, threshold=0.35):
    start = time.time()
    results = retriever.invoke(query)
    elapsed = time.time() - start
    retrieved = [doc.page_content for doc in results[:10]]
    prec, rec, f1 = evaluate_accuracy(retrieved, expected, embedding_model, threshold)
    return elapsed, prec, rec, f1, retrieved

def evaluate_answer(query, expected, chain):
    try:
        start = time.time()
        result = chain.invoke({"query": query})
        elapsed = time.time() - start
        answer = result["result"]
        sim = max(cosine_sim(answer, e, embedding) for e in expected)
        return elapsed, answer, sim
    except Exception as e:
        return 0, str(e), 0

# === 실행 === #
if __name__ == "__main__":
    testset = [
        {
            "query": "학습 전략의 종류는 무엇인가요?",
            "expected": [
                "학습 전략에는 인지 전략, 메타인지 전략, 자원 관리 전략이 포함됩니다.",
                "메타인지 전략은 자신의 학습을 계획, 모니터링, 평가하는 전략입니다."
            ]
        },
        {
            "query": "학생의 자기조절 학습 방법에 대해 설명해줘.",
            "expected": [
                "자기조절 학습은 목표 설정, 자기 모니터링, 자기 평가를 포함합니다.",
                "학생들은 목표를 설정하고 학습 과정 중 자신을 점검하며 평가합니다."
            ]
        }
    ]

    for i, case in enumerate(testset, 1):
        print(f"\n🧪 테스트 {i}: {case['query']}")

        db_time, prec, rec, f1, retrieved = evaluate_query(case["query"], case["expected"], embedding, retriever)
        print(f"🔍 검색 시간: {db_time:.2f}s | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

        ans_time, answer, sim = evaluate_answer(case["query"], case["expected"], qa_chain)
        print(f"🧠 응답 시간: {ans_time:.2f}s | 유사도: {sim:.3f}\n📝 답변 요약: {answer[:100]}...")
