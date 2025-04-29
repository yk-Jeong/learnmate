import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

# 설정
SAVE_PDF_DIR = "/Users/jeong/AI/learnmate/data/papers"
SAVE_MD_DIR = "/Users/jeong/AI/learnmate/data/abstracts"
KEYWORD = "learning strategy of secondary school student OR metacognition"
MAX_RESULTS = 3
TITLE_SLICE = 60

# 파일명 정리
def sanitize_filename(text):
    return "".join(c for c in text if c.isalnum() or c in " ._-").rstrip()

# 논문 가져오기
def fetch_arxiv(keyword=KEYWORD, max_results=MAX_RESULTS):
    os.makedirs(SAVE_PDF_DIR, exist_ok=True)
    os.makedirs(SAVE_MD_DIR, exist_ok=True)

    print(f"📡 arXiv에서 '{keyword}' 검색 중...")

    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{keyword}",
        "start": 0,
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
    print(f"🔍 총 {len(entries)}편 논문 발견\n")

    for entry in entries:
        try:
            title = entry.find("atom:title", ns).text.strip()
            summary = entry.find("atom:summary", ns).text.strip()
            pdf_url = entry.find("atom:id", ns).text.strip().replace("abs", "pdf")

            filename_base = sanitize_filename(title[:TITLE_SLICE])
            md_path = os.path.join(SAVE_MD_DIR, f"{filename_base}.md")
            pdf_path = os.path.join(SAVE_PDF_DIR, f"{filename_base}.pdf")

            # Markdown 저장
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n")
                f.write(f"**Abstract:**\n{summary}\n\n")
                f.write(f"[arXiv 원문 링크]({pdf_url})\n")
            print(f"📝 요약 저장 완료: {md_path}")

            # PDF 다운로드 및 삭제
            pdf_res = requests.get(pdf_url + ".pdf")
            pdf_res.raise_for_status()

            with open(pdf_path, "wb") as f:
                f.write(pdf_res.content)
            print(f"✅ PDF 저장 완료: {pdf_path}")

            os.remove(pdf_path)
            print(f"🗑 PDF 삭제 완료: {pdf_path}\n")

        except Exception as e:
            print(f"⚠️ 논문 '{title}' 처리 중 오류: {e}\n")

if __name__ == "__main__":
    print(f"⏱ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    fetch_arxiv()
import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

# 설정
SAVE_PDF_DIR = "/Users/jeong/AI/learnmate/data/papers"
SAVE_MD_DIR = "/Users/jeong/AI/learnmate/data/abstracts"
KEYWORD = "learning strategy of secondary school student OR metacognition"
MAX_RESULTS = 3
TITLE_SLICE = 60

# 파일명 정리
def sanitize_filename(text):
    return "".join(c for c in text if c.isalnum() or c in " ._-").rstrip()

# 논문 가져오기
def fetch_arxiv(keyword=KEYWORD, max_results=MAX_RESULTS):
    os.makedirs(SAVE_PDF_DIR, exist_ok=True)
    os.makedirs(SAVE_MD_DIR, exist_ok=True)

    print(f"📡 arXiv에서 '{keyword}' 검색 중...")

    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{keyword}",
        "start": 0,
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
    print(f"🔍 총 {len(entries)}편 논문 발견\n")

    for entry in entries:
        try:
            title = entry.find("atom:title", ns).text.strip()
            summary = entry.find("atom:summary", ns).text.strip()
            pdf_url = entry.find("atom:id", ns).text.strip().replace("abs", "pdf")

            filename_base = sanitize_filename(title[:TITLE_SLICE])
            md_path = os.path.join(SAVE_MD_DIR, f"{filename_base}.md")
            pdf_path = os.path.join(SAVE_PDF_DIR, f"{filename_base}.pdf")

            # Markdown 저장
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n")
                f.write(f"**Abstract:**\n{summary}\n\n")
                f.write(f"[arXiv 원문 링크]({pdf_url})\n")
            print(f"📝 요약 저장 완료: {md_path}")

            # PDF 다운로드 및 삭제
            pdf_res = requests.get(pdf_url + ".pdf")
            pdf_res.raise_for_status()

            with open(pdf_path, "wb") as f:
                f.write(pdf_res.content)
            print(f"✅ PDF 저장 완료: {pdf_path}")

            os.remove(pdf_path)
            print(f"🗑 PDF 삭제 완료: {pdf_path}\n")

        except Exception as e:
            print(f"⚠️ 논문 '{title}' 처리 중 오류: {e}\n")

if __name__ == "__main__":
    print(f"⏱ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    fetch_arxiv()
