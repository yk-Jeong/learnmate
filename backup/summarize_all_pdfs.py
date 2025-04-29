import os
import pdfplumber
from openai import OpenAI
from textwrap import wrap
from dotenv import load_dotenv
from datetime import datetime

# 경로 설정
PDF_DIR = "/Users/jeong/AI/learnmate/data/papers"
SUMMARY_DIR = "/Users/jeong/AI/learnmate/data/abstracts"

# 환경 변수에서 OpenAI API 키 불러오기
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 텍스트 추출
def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"❌ PDF 로딩 오류 ({pdf_path}): {e}")
        return ""

# 텍스트를 문자수 기준으로 쪼개기
def chunk_text(text, max_chars=4000):
    return wrap(text, width=max_chars)

# 단일 chunk 요약
def summarize_single_chunk(chunk):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "너는 교육 관련 논문을 요약하는 한국어 전문가야. 핵심을 조리 있게 요약해줘."},
                {"role": "user", "content": f"다음 내용을 요약해줘:\n{chunk}"}
            ],
            max_tokens=800,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 요약 실패: {e}")
        return ""

# 전체 문서 요약
def summarize_text(text):
    chunks = chunk_text(text)
    partial_summaries = []

    # 1차 chunk 요약
    for idx, chunk in enumerate(chunks):
        print(f"✂️ Chunk {idx+1}/{len(chunks)} 요약 중...")
        partial_summary = summarize_single_chunk(chunk)
        if partial_summary:
            partial_summaries.append(partial_summary)

    combined_summary = "\n".join(partial_summaries)

    # 2차 최종 요약
    print("\n🌀 2차 최종 요약 중...")
    final_summary = summarize_single_chunk(combined_summary)
    return final_summary

# 요약 저장
def save_summary_to_md(title, summary):
    filename = "".join(c for c in title[:60] if c.isalnum() or c in " ._-").rstrip().replace(" ", "_") + ".md"
    path = os.path.join(SUMMARY_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{summary}\n")
    print(f"✅ 요약 저장 완료: {path}")

# 전체 파이프라인
def run_pipeline():
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    for filename in os.listdir(PDF_DIR):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        print(f"\n📄 논문 처리 시작: {filename} ({datetime.now().strftime('%H:%M:%S')})")

        text = extract_text_from_pdf(pdf_path)
        if not text:
            continue

        summary = summarize_text(text)
        if not summary:
            continue

        save_summary_to_md(filename.replace(".pdf", ""), summary)

        os.remove(pdf_path)
        print(f"🗑 PDF 삭제 완료: {pdf_path}")

# 실행
if __name__ == "__main__":
    print(f"⏱ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run_pipeline()
