## personal project: Learnmate
### ✅ Abstract

- 목표: 사용자의 학습 고민을 자연어로 입력 → RAG 기법을 통해 실제 교육학 논문 자료를 바탕으로 한 심리적·전략적 조언을 생성 → **학생 개인의 학습 성향에 맞춤형 학습 전략을 제공**함으로써, 자기주도 학습을 지원하는 챗봇 구현
- 데이터: eric에서 수집한 학습 전략 및 상위인지 관련 논문의 요약문

### 🧪 Method

1. 기술 스택
    
    
    | 구성 요소 | 도구 |
    | --- | --- |
    | 문헌 전처리 | PyPDF2, pdfplumber |
    | Sentence embedding model | intfloat/multilingual-e5-large |
    | Vector DB | FAISS |
    | RAG framework | LangChain |
    | LLM API  | OpenAI GPT-4o, gpt-4.1  |
    | UI framework | Streamlit |
2. **데이터 구성**
    
    
    | 출처 | 원본 데이터 | 활용 |
    | --- | --- | --- |
    | eric | 학습 심리 및 상위인지 관련 논문 PDF | 요약 후 임베딩 및 RAG 검색 |

### 🔄 Process

1. **데이터 수집 및 전처리**
    - ERIC(교육학 전문 논문 포털)에서 학습전략/상위인지 관련 논문 스크래핑 → PDF에서 텍스트 추출 → LLM(GPT-4.1)로 요약 → markdown(.md) format으로 데이터베이스화
2. **RAG 파이프라인 구성**
    - intfloat/multilingual-e5-large 기반 임베딩 생성 → FAISS에 저장, 벡터 기반 검색 인프라 구축 → LangChain을 이용해 GPT-4.1과 연동된 RetrievalQA 체인 구성
3. **성능 평가**
    - 정확도(정답 포함 여부)
    - F1 score(재현율 및 정밀도 기반)
    - LLM 응답의 유사도 및 응답시간
    - 유사 문서 검색 품질 측정(코사인 유사도 기반)

### 📊 Result & Limitation

- 최종 성능:
- 

### ➕ Futhermore

- **사용자가 직접 논문을 수집하는 기능 추가**
- **학습자 유형 진단 테스트 연동** (Kolb, Felder 등)
- **교사/학부모 전용 모드** 제공
- **성장 일지 및 리포트 트래킹**(학습자 고민, 제안한 전략, 피드백 이력 저장)
- **멀티모달 확장**(TTS / 이미지 내 텍스트 OCR 기반 질의 등)
