# 💳 Card Benefits Chatbot (Streamlit + LangChain)

**“혜택을 읽고, 사용자 맥락으로 추천합니다.”**  
로컬 JSON(`data/card_llm_ready.json`)을 임베딩/검색하여 **카드 혜택 질의·추천**을 제공하는 경량 챗봇입니다.  
**LangChain + FAISS + (옵션) CrossEncoder 리랭커 + Ollama/HF LLM** 조합으로 **빠른 응답 · 낮은 운영비 · 쉬운 확장**을 목표로 설계했습니다.

> UI: `streamlit_web.py` · 체인/로직: `main.py` · 데이터: `data/card_llm_ready.json`  
> 크롤링(옵션): `tools/crawling.py` (Selenium, 대상 사이트 정책 준수 권장)

---

## 🧭 Executive Summary (for Stakeholders)
- **Problem**: 혜택이 복잡·분절되어 사용자가 “내 상황”에 맞는 카드를 고르기 어렵다.
- **Solution**: 로컬 JSON → 임베딩 검색 + (옵션) 리랭크/LLM 요약으로 **맥락 기반 Top‑N 추천**.
- **Differentiators**
  - ⚡ **로컬‑우선 아키텍처**: 외부 의존 최소 → 낮은 비용/짧은 응답/쉬운 이식
  - 🔎 **투명한 추천 근거**: 매칭 키워드/설명 근거 노출 → 신뢰성/디버깅 용이
  - 🧱 **모듈형 확장**: 크롤링·룰 엔진·LLM 교체·배치 갱신 등 단계적 확장 용이

---

## ✨ Features
- **자연어 + 키워드 질의**: “지하철/버스/교통” 등 유사 표현을 **표준 키워드**로 정규화
- **의미 기반 검색**: 혜택 설명 임베딩 + (옵션) 리랭킹으로 **정확도 향상**
- **추천 사유 제공**: 매칭 키워드/설명 일부를 함께 반환 → 투명성/디버깅 용이
- **로컬‑친화**: 파일 데이터로 시작 → 필요 시 DB/API/배치 갱신으로 확장

---

## 🧱 Prompt Engineering

### 1) 역할·출력 포맷(시스템 프롬프트)
모델을 **신용카드 혜택 전문가**로 역할 고정하고, 출력 포맷을 **JSON 스키마**로 고정합니다.
```text
You are a credit-card benefit expert. Answer concisely with evidence.
Rules:
- Return JSON with fields: intent, reasons[], suggestions[]
- Show matched keywords and short evidence snippets when possible
- If uncertain, ask for one clarifying preference
```

### 2) 카테고리 라우팅(추천/비교/Q&A/기타)
질의를 4유형으로 분류하여 **각기 다른 체인**으로 라우팅합니다.
```text
[intents] = { "recommend", "compare", "benefit_qa", "other" }
Choose the most specific intent. Prefer NOT "other" if any card-benefit intent fits.
```
- **Few‑shot 예시**를 각 intent에 1~2개 포함하여 경계 사례(예: Q&A↔기타 혼동)를 줄입니다.

### 3) 근거 중심 요약 프롬프트
검색 결과(카드 혜택 설명)에서 **증거 문장**을 추출해 2~3줄 요약으로 정리합니다.
```text
Summarize top benefits in 2-3 bullet lines. Cite which keywords matched: ["교통","편의점",...]
```

---

## 🧩 LangChain Orchestration

### 체인 토폴로지
- **Embedding → FAISS**: `혜택 설명`을 단락으로 분할 후 임베딩 저장
- **Retriever**: 질의(정규화된 키워드 포함)를 인코딩 → 유사도 Top‑K 문서 반환
- **(옵션) Re‑rank**: CrossEncoder로 상위 후보 재정렬
- **LLM Summarizer**: 근거 포함 요약/추천 사유 생성
- **Memory**: `ConversationBufferMemory`로 최근 대화를 **히스토리**로 유지

### 예시 코드 스케치
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vs = FAISS.from_texts(texts, emb)

retriever = vs.as_retriever(search_kwargs={"k": 8})
memory = ConversationBufferMemory(k=4, return_messages=True)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a credit-card benefit expert..."),
    ("human", "{question}")
])

def re_rank(docs):  # optional CrossEncoder
    return docs  # placeholder

chain = (
    {"docs": retriever | re_rank, "question": RunnablePassthrough()}
    | prompt
    # | 모델 호출 (Ollama/HF)
)
```

### History(대화 히스토리)
- 최근 **k=4 turns** 유지(토큰 초과 방지), 필요 시 요약 압축.
- 히스토리는 **의도/선호 유지**에만 사용, 응답 본문 근거는 **검색 결과**에서만 추출.

---

## 🌐 Tavily Web Search (옵션)
외부 최신 정보가 필요한 비교/이슈성 질의에만 사용합니다.
- **키**: `TAVILY_API_KEY` (환경변수)
- **모드**: 일반/뉴스/이미지, 안전·도메인 필터링 옵션
- **온오프**: `.env`로 끄고 켤 수 있게 설계
- **머지**: 검색 결과 요약을 **근거 블록**에 병합하여 표기

```python
from tavily import TavilyClient
tc = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))
resp = tc.search(query, search_depth="advanced", max_results=5)
```

---

## 🏗 Architecture
```text
[Browser/Streamlit]
      │  질의(선호/맥락)
      ▼
[streamlit_web.py]  ──►  [main.py Orchestrator]
                          ├─ FAISS VectorStore (임베딩 검색)
                          ├─ (opt) CrossEncoder Re-ranker
                          ├─ ConversationBufferMemory (History)
                          ├─ (opt) Tavily Web Search
                          └─ Ollama/HF LLM (요약/정리)
                              ▼
                        Ranked Results + Reasons
```

---

## 📂 Repository Structure
아래 구조는 현재 레포의 실제 파일 기준입니다.
```
.
├─ data/
│  └─ card_llm_ready.json        # 카드 혜택 JSON (UTF-8)
├─ tools/
│  └─ crawling.py                # 혜택 데이터 수집(옵션, Selenium)
├─ main.py                       # 체인: 임베딩/검색/리랭크/요약 오케스트레이션
├─ streamlit_web.py              # Streamlit UI
├─ model.ipynb                   # (옵션) 실험/모델링 노트북
├─ test.ipynb                    # (옵션) 테스트/데모 노트북
├─ requirements.txt
├─ .env.sample                   # TAVILY_API_KEY, LLM_MODEL, CARD_DATA 등
├─ .gitignore
└─ Dockerfile                    # 컨테이너 실행(8501)
```

---

## ⚙️ Setup
```bash
git clone https://github.com/<YOUR_ID>/card-chatbot.git
cd card-chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
cp .env.sample .env   # 필요 시 TAVILY_API_KEY / LLM_MODEL / CARD_DATA 설정
```
> 로컬 LLM(Ollama) 사용 시 시스템에 Ollama 설치 및 모델 pull 필요(예: `mistral:latest`).

---

## ▶️ Run
```bash
streamlit run streamlit_web.py
# 브라우저: http://localhost:8501
```
- 선호 카테고리(예: 교통/통신/편의점) 입력 → **추천 결과 + 매칭 근거** 확인  
- `main.py` 체인이 **정규화 → 검색/리랭크 → 요약**을 수행합니다.

---

## 🔌 Config
| Key              | Default                    | Description                  |
|------------------|----------------------------|------------------------------|
| `CARD_DATA`      | `data/card_llm_ready.json` | 카드 혜택 JSON 경로          |
| `TAVILY_API_KEY` | *(empty)*                  | (옵션) 웹 검색 기능 키       |
| `LLM_MODEL`      | `mistral:latest`           | Ollama/HF 모델명(옵션)       |

> 비밀키/환경설정은 `.env`에서 관리하세요(커밋 금지).

---

## 🔎 Data Schema (excerpt)
```json
[
  {
    "카드 이름": "예시 카드",
    "카드 회사": "ABC",
    "혜택 키워드": "대중교통",
    "혜택 설명": "버스/지하철 10% 할인 (월 최대 1만원)"
  }
]
```
- **키워드 예시**: 교통/주유/통신/해외/편의점/카페/영화/쇼핑/구독/마일리지/공과금 …  
- “지하철/버스/교통” → `교통` 등 **정규화/동의어 매핑** 적용

---

## 📈 Business Impact & Metrics
| Metric | 정의 | 측정 방법 |
|---|---|---|
| **Top‑3 Hit@K** | 추천 Top‑3 중 사용자 선택 카드 포함 비율 | UI 선택 로그/이벤트로 히트율 산출 |
| **Time‑to‑Answer** | 질의→결과 렌더까지 응답 시간(ms) | Streamlit 요청/응답 타이머 |
| **Explainability CTR** | “추천 근거 펼침” 클릭률 | 근거 섹션 토글 이벤트 |

```python
# (예시) metrics.py – 간단 로깅 훅
from time import perf_counter
def with_timer(fn):
    def wrap(*a, **kw):
        t0 = perf_counter(); r = fn(*a, **kw); dt = int((perf_counter()-t0)*1000)
        print({"metric":"time_to_answer_ms","value":dt})
        return r
    return wrap
```

---

## 🔐 Security & Privacy
- **비저장 모드 기본값**: 사용자 질의/응답 **서버 저장 없음**(옵션으로 익명 통계만)
- **비밀키 관리**: `.env` 환경변수, 저장소 커밋 금지
- **외부 호출 제어**: 기본 **로컬‑온리**, 실시간 검색(Tavily)은 **옵션** 플래그

---

## 🛠️ Production Readiness
- **스케일링**: 컨테이너(🐳 Docker) → Streamlit Cloud/Render/Cloud Run
- **관측성**: 시간/히트율/에러 로그 표준화 → 대시보드 연결(예: Grafana)
- **룰 엔진**(Roadmap): 전월 실적/상한/중복 할인 **가중치화**
- **브랜드 표준화**(Roadmap): “스타벅스→카페”, “이마트→대형마트” 상위 카테고리 매핑

---

## 🧪 Quality Gates (권장)
- **pytest**: 정규화/퍼지매칭/점수 계산 단위 테스트
- **데이터 검증**: JSON 스키마 검사(필수 필드/한글 인코딩/중복 라인)
- **프롬프트 가드**: 카테고리 라우팅(추천/비교/Q&A/기타) few‑shot 예시 유지

---

## 🧭 Roadmap
- [ ] 실적/한도 룰 엔진(상한·페널티) → 점수 가중치 구조화  
- [ ] 브랜드 매핑(스타벅스→카페, 이마트→대형마트)  
- [ ] 개인화(최근 소비 카테고리 Top‑N 가중치)  
- [ ] 배포 가이드(Streamlit Cloud/Render/Cloud Run) & CI/CD(테스트/릴리즈)  
- [ ] 관측성 강화(추천 지표 대시보드)

---

## 🐳 Docker (옵션)
```bash
docker build -t card-chatbot .
docker run -it --rm -p 8501:8501 \
  -e CARD_DATA=data/card_llm_ready.json \
  --name card-bot card-chatbot
# http://localhost:8501
```

---

## License
MIT (또는 내부 배포 정책에 맞춰 변경)
