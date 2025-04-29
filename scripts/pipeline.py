import os
import requests
import xml.etree.ElementTree as ET
import pdfplumber
from datetime import datetime
from openai import OpenAI
from textwrap import wrap
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# 설정
# =========================
PDF_DIR = "/Users/jeong/AI/learnmate/data/papers"
SUMMARY_DIR = "/Users/jeong/AI/learnmate/data/abstracts"
VECTOR_STORE_DIR = "/Users/jeong/AI/learnmate/vectordb"

KEYWORD = "learning strategy of student"
START_INDEX = 0 
MAX_RESULTS = 50
TITLE_SLICE = 60
CHUNK_SIZE = 3000

# =========================
# 환경 설정
# =========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# 함수 정의
# =========================

def sanitize_filename(text):
    return "".join(c for c in text if c.isalnum() or c in " ._-").rstrip()

def fetch_arxiv(keyword=KEYWORD, start_index=START_INDEX, max_results=MAX_RESULTS):
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    print(f"📡 arXiv에서 '{keyword}' 검색 중...")

    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"title:{keyword}",
        "start": start_index,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    try:
        res = requests.get(base_url, params=params)
        res.raise_for_status()
    except Exception as e:
        print(f"❌ arXiv 요청 실패: {e}")
        return

    root = ET.fromstring(res.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    entries = root.findall("atom:entry", ns)
    print(f"🔍 {len(entries)}편 논문 검색됨\n")

    for entry in entries:
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip()
        pdf_url = entry.find("atom:id", ns).text.strip().replace("abs", "pdf")

        filename_base = sanitize_filename(title[:TITLE_SLICE])
        md_path = os.path.join(SUMMARY_DIR, f"{filename_base}.md")
        pdf_path = os.path.join(PDF_DIR, f"{filename_base}.pdf")

        # 요약 저장
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(f"**Abstract:**\n{summary}\n\n")
            f.write(f"[arXiv 원문 링크]({pdf_url})\n")
        print(f"📝 요약 저장 완료: {md_path}")

        # PDF 다운로드
        try:
            pdf_res = requests.get(pdf_url + ".pdf")
            pdf_res.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(pdf_res.content)
            print(f"✅ PDF 저장 완료: {pdf_path}")
        except Exception as e:
            print(f"❌ PDF 다운로드 실패: {e}")

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"❌ PDF 로딩 오류 ({pdf_path}): {e}")
        return ""

def chunk_text(text, max_chars=CHUNK_SIZE):
    return wrap(text, width=max_chars)

def summarize_text(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "너는 교육 관련 논문을 요약하는 한국어 전문가야. 핵심을 명확히 요약해줘."},
                {"role": "user", "content": f"다음 논문을 요약해줘:\n{text}"}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 최종 요약 실패: {e}")
        return ""

def save_summary_to_md(title, summary):
    filename = "".join(c for c in title[:TITLE_SLICE] if c.isalnum() or c in " ._-").rstrip().replace(" ", "_") + ".md"
    path = os.path.join(SUMMARY_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{summary}\n")
    print(f"✅ 최종 요약 저장 완료: {path}")

# =========================
# 메인 파이프라인
# =========================

def run_pipeline():
    fetch_arxiv()

    all_docs = []

    for filename in os.listdir(PDF_DIR):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        print(f"\n📄 논문 처리 시작: {filename}")

        text = extract_text_from_pdf(pdf_path)
        if not text:
            continue

        # Chunk 분할 → DB 저장용
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            doc = Document(page_content=chunk, metadata={"source": filename, "chunk_idx": idx})
            all_docs.append(doc)

        # 최종 요약 → .md 저장
        summary = summarize_text(text)
        if summary:
            save_summary_to_md(filename.replace(".pdf", ""), summary)

        # PDF 삭제
        os.remove(pdf_path)
        print(f"🗑 PDF 삭제 완료: {pdf_path}")

    # 모든 문서 FAISS 저장
    if all_docs:
        print("\n🧠 모든 chunk를 벡터화 중...")
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(all_docs, embedding)
        db.save_local(VECTOR_STORE_DIR)
        print(f"✅ 벡터 DB 저장 완료: {VECTOR_STORE_DIR}")

# 실행
if __name__ == "__main__":
    print(f"⏱ 전체 파이프라인 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    run_pipeline()
