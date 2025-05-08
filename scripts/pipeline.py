import os
import time
import requests
import pdfplumber
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from textwrap import TextWrapper

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# 설정
# =========================
BASE_URL = "https://eric.ed.gov"
SEARCH_URL = "https://eric.ed.gov/?q=learning+strategy+OR+study+skill+OR+metacognition+of+student&ft=on&ff1=dtySince_2024"

PDF_DIR = "data/papers"
SUMMARY_DIR = "data/abstracts"
VECTOR_STORE_DIR = "vectordb"
TITLE_SLICE = 60
CHUNK_SIZE = 5000  # Max chunk size in tokens

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

def fetch_paper_links(max_pages=10):
    print(f"🔍 ERIC 논문 목록 가져오는 중...")
    paper_links = []
    page_num = 0
    next_url = SEARCH_URL

    while next_url and page_num < max_pages:
        page_num += 1
        print(f"📄 페이지 {page_num} 요청 중: {next_url}")
        try:
            res = requests.get(next_url)
            res.raise_for_status()
        except Exception as e:
            print(f"⚠️ 요청 실패 (페이지 {page_num}): {e}")
            break

        soup = BeautifulSoup(res.text, "html.parser")

        entries = soup.select(".r_i")
        links = [urljoin(BASE_URL, entry.select_one("a")["href"]) for entry in entries if entry.select_one("a")]
        paper_links.extend(links)

        next_button = soup.find("a", string="Next Page »")
        if next_button and next_button.get("href"):
            next_url = urljoin(BASE_URL, next_button["href"])
        else:
            next_url = None

        time.sleep(1)

    print(f"📚 총 {len(paper_links)}편 논문 수집 완료 (최대 {max_pages}페이지 기준)")
    return paper_links



def download_pdf(paper_url):
    eric_id = paper_url.split("id=")[-1]
    title = eric_id
    pdf_url = f"http://files.eric.ed.gov/fulltext/{eric_id}.pdf"
    filename = sanitize_filename(title[:TITLE_SLICE]) + ".pdf"
    path = os.path.join(PDF_DIR, filename)

    if not os.path.exists(path):
        print(f"⬇️ 다운로드 중: {eric_id}")
        try:
            pdf_res = requests.get(pdf_url)
            pdf_res.raise_for_status()
            with open(path, "wb") as f:
                f.write(pdf_res.content)
            print(f"✅ 저장 완료: {path}")
            return eric_id, path
        except Exception as e:
            print(f"❌ PDF 다운로드 실패 ({eric_id}): {e}")
            return None, None
    else:
        print(f"📦 이미 존재함: {path}")
        return eric_id, path

def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages).strip()
    except Exception as e:
        print(f"❌ PDF 로딩 실패: {e}")
        return ""

def chunk_text(text, max_tokens=CHUNK_SIZE):
    # Estimate token count by considering a rough average of 4 characters per token
    max_chars = max_tokens * 4  # approximate
    wrapper = TextWrapper(width=max_chars, break_long_words=False)
    return wrapper.wrap(text)

def summarize_text(text):
    try:
        # Split text into smaller chunks based on token length
        chunks = chunk_text(text)
        summaries = []
        
        for chunk in chunks:
            # Check token length for each chunk to avoid exceeding the limit
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "너는 교육 관련 논문을 요약하는 한국어 전문가야. 핵심 내용을 요약해줘."},
                    {"role": "user", "content": f"다음 논문을 10000단어 내외로 요약:\n{chunk}"}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            summaries.append(response.choices[0].message.content.strip())
        
        return " ".join(summaries)  # Combine all summaries from chunks
    except Exception as e:
        print(f"❌ 요약 실패: {e}")
        return ""

def save_summary_to_md(title, summary):
    filename = sanitize_filename(title[:TITLE_SLICE]) + ".md"
    path = os.path.join(SUMMARY_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{summary}\n")
    print(f"✅ 요약 저장 완료: {path}")
    return path

def run_pipeline():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    links = fetch_paper_links(max_pages=10)
    all_docs = []

    for link in links:
        try:
            title, pdf_path = download_pdf(link)
            if not pdf_path:
                continue

            # 요약 .md 파일이 이미 존재하면 스킵
            md_filename = sanitize_filename(title[:TITLE_SLICE]) + ".md"
            md_path = os.path.join(SUMMARY_DIR, md_filename)
            if os.path.exists(md_path):
                print(f"⏩ 이미 요약됨: {md_path}")
                continue

            text = extract_text_from_pdf(pdf_path)
            os.remove(pdf_path)
            print(f"🗑 PDF 삭제 완료: {pdf_path}")

            if not text:
                print(f"⏩ 텍스트 없음: {title}")
                continue

            summary = summarize_text(text)
            if not summary:
                continue

            save_summary_to_md(title, summary)
            chunks = chunk_text(summary)

            for idx, chunk in enumerate(chunks):
                doc = Document(page_content=chunk, metadata={"source": md_path, "chunk_idx": idx})
                all_docs.append(doc)

            time.sleep(2)

        except Exception as e:
            print(f"❌ 처리 실패: {e}")
            continue

    if all_docs:
        print("\n🧠 모든 요약을 벡터화 중...")
        embedding = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))
        db = FAISS.from_documents(all_docs, embedding)
        db.save_local(VECTOR_STORE_DIR)
        print(f"✅ 벡터 DB 저장 완료: {VECTOR_STORE_DIR}")
    else:
        print("📭 벡터화할 문서가 없습니다.")

# 실행
if __name__ == "__main__":
    print(f"\n⏱ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run_pipeline()
