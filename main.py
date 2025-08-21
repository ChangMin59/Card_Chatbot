import os
import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder # 리랭커
from langchain_community.tools import TavilySearchResults
from langchain.llms import HuggingFacePipeline
#from langchain_community.llms import HuggingFacePipeline
# for huggingface
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")


# === 카드 데이터 로드 ===
DATA_PATH = os.getenv("CARD_DATA", "data/card_llm_ready.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# === 카드 문서 변환 ===
docs = [
    Document(
        page_content=f"[{item['카드 이름']}] - {item['카드 회사']} ({item['혜택 키워드']}) 혜택: {item['혜택 설명']}",
        metadata={
            "카드 이름": item["카드 이름"],
            "카드 회사": item["카드 회사"],
            "혜택 키워드": item["혜택 키워드"],
            "혜택 설명": item["혜택 설명"]
        }
    ) for item in raw_data
]

# === 임베딩 및 벡터스토어 설정 ===
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

#if os.path.exists("faiss_index"):
#    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
#else:
vectorstore = FAISS.from_documents(docs, embedding_model)
vectorstore.save_local("faiss_index")


# 리랭커: 검색기(retriever) 설정. 문서수 3123개 (/home/alpaco/1t/changmin/card_llm_ready.json)
num_docs = 15     # 2 ~ 10 개, 기존 main.py 코드에서 k가 15였으므로 15로 변경 10 -> 15.
k = num_docs * 5  # 10 ~ 50 개, 속도가 느릴 경우 줄일 수 있음.
fetch_k = k * 10   # k비중 2~10 배
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k, "fetch_k": fetch_k})
retriever0 = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 15})

# 리랭커: 모델 초기화
reranker_model = CrossEncoder('Dongjin-kr/ko-reranker')

# 리랭커: 리트리버에 리랭커를 결합한 검색 함수
def advanced_retrieval(query):
    # 초기 유사도 검색
    initial_results = retriever.invoke(query)

    if not initial_results:
        return []
    
    # 리랭커를 사용한 재정렬: query(질문), page_content(리트리버가 가져온 doc)
    pairs = [[query, doc.page_content] for doc in initial_results]
    reranked_scores = reranker_model.predict(pairs)
    
    #점수에 따라 문서 재정렬
    reranked_docs = [
        doc for _, doc in sorted(
            zip(reranked_scores, initial_results), 
            key=lambda x: x[0], 
            reverse=True
        )
    ]

    return reranked_docs[:num_docs]  # 상위 문서만 반환

# 리랭커: 검색함수 할당
reranker = RunnableLambda(advanced_retrieval)


# === 문맥 포맷 함수 ===
def format_docs(docs):
    if not docs:
        return "관련된 카드를 찾지 못했어요."
    seen = set()
    formatted = []
    for i, doc in enumerate(docs, 1):
        name = doc.metadata["카드 이름"]
        company = doc.metadata["카드 회사"]
        benefit = doc.page_content.split("혜택: ")[-1] if "혜택: " in doc.page_content else "혜택 정보 없음"
        if name not in seen:
            seen.add(name)
            formatted.append(f"{i}. {name} ({company}) - 혜택: {benefit}")
    return "\n".join(formatted)

# === 대화 메모리 ===
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="output",
    return_messages=True
)

def get_chat_history(_):
    return "\n".join(
        [f"{msg.type.capitalize()}: {msg.content}" for msg in memory.load_memory_variables({})["chat_history"]]
    )

# === LLM 로딩 ===
llm_model = os.getenv("LLM_MODEL", "mistral:latest")
llm = Ollama(model=llm_model)
merged_model_path = "./aris"
model = AutoModelForCausalLM.from_pretrained(merged_model_path, device_map="auto",torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    #temperature=0.7,
    #top_p=0.9,
    #repetition_penalty=1.1,
    truncation=True,
    do_sample=False,
    use_cache=True
)
llm_aris = HuggingFacePipeline(pipeline=pipe)

# 질문 분류 프롬프트
classification_prompt = ChatPromptTemplate.from_template(
    """
다음 사용자 질문이 어떤 목적에 해당하는지 판단해줘.
총 4가지 유형이 있어. 설명과 예시를 잘 참고해서 가장 적절한 번호 하나만 골라줘.

※ 답변은 반드시 숫자 하나만 줘 (예: 1). 설명이나 이유는 절대 쓰지 마.
※ 가능한 경우 1~3번 중에서 선택하려고 노력해줘.
※ 4번은 카드랑 정말 관련 없는 잡담일 때만 골라줘.

[카테고리 유형]

1. 특정 상황이나 대상에 맞는 카드 추천  
→ 사용자가 본인의 상황(예: 주유소 자주 이용, 온라인 쇼핑 자주 함 등)을 말하며  
  카드 종류를 어느 정도 알고 있는 경우  
→ 사용자가 "이런 상황에 맞는 카드 알려줘"라는 식으로 질문함

예시:
- 주유소 할인되는 카드 알려줘  
- 20대 여성 직장인에게 좋은 카드 추천해줘  
- 카페 자주 가는 사람에게 혜택 좋은 카드 뭐 있어?  
- 병원 자주 다니는 사람에게 유리한 카드 있어?  
- 여행 자주 가는 사람에게 좋은 카드 추천해줘  

2. 특정 카드 혜택 조회 또는 카드 간 비교  
→ 카드 이름이 질문에 직접적으로 등장하며  
  해당 카드에 대한 정보나 비교를 요청하는 경우

예시:
- 삼성카드 the O 혜택 뭐야?  
- 현대 the Green이랑 신한 Deep Oil 비교해줘  
- Deep Oil이랑 Deep On 카드 차이점은 뭐야?  
- 이디야 할인되는 카드가 뭐야?  

3. 사용자 맞춤 카드 추천 (역질문 필요)  
→ 사용자가 본인의 상황을 구체적으로 말하지 않거나  
  어떤 카드를 선택해야 할지 모르는 상태  
→ 이 경우에는 성별, 나이, 관심사 등을 되물어야 하므로  
  **역질문이 필요한 경우로 판단**

예시:
- 나한테 맞는 카드 추천해줘  
- 어떤 카드가 나랑 잘 맞을까?  
- 카드 하나 만들까 하는데 뭐가 좋을까?  
- 요즘 카드 뭐가 잘 나가? 나한테 어울리는 거 추천해줘  
- 신용카드 처음 만들건데 어디서부터 시작해야 할까?  
- 아무 카드나 쓰기 싫은데 나한테 맞는 카드가 필요해  
- 신용카드 잘 몰라서 추천해줘  
- 상황에 맞는 걸 추천해줬으면 좋겠어  

4. 카드와 무관한 질문 (잡담 등)  
→ 신용카드와 관련 없는 질문 또는 일반적인 대화

예시:
- 오늘 날씨 어때?  
- 넌 누구야?  
- 심심한데 뭐하지?  
- 밥 뭐 먹을까?  
- 너 몇 살이야?  

질문: {question}  
답변 (숫자 하나만):
"""
)

classifier_chain = classification_prompt | llm | StrOutputParser()

# 각각의 응답 방식 정의
# 1번 응답: 기본 카드 추천 방식
card_recommend_template = ChatPromptTemplate.from_template("""
너는 한국어로 신용카드 혜택을 친절하게 설명하는 전문가야.

- 처음에는 사용자의 질문에 관련된 신용카드 목록을 보여줘. 각 카드별로 이름과 주요 혜택을 간략하게 요약해줘.
- 사용자가 원하는 카드 개수를 말하지 않았다면 최대한 다양한 옵션을 보여줘.
- 카드명은 context에 등장한 실제 카드명만 사용하고, 간단하고 명확한 표현으로 요약해줘.
- 리스트가 끝나면, 그 안에서 **가장 혜택이 좋은 카드 1~2개만 추천**해
- 마지막에는 혜택 비교 및 추천해줘 추천한 기준도 간단히 정리해줘

# 추천 신용카드 목록:
{context}

# 이전 대화:
{chat_history}

# 질문:
{question}

# 답변:
""")
recommend_chain = {
        "context": reranker | RunnableLambda(format_docs), # 리랭커: retriever0 -> reranker
        "chat_history": RunnableLambda(get_chat_history),
        "question": RunnablePassthrough(),
} | card_recommend_template | llm

# 2번 응답: 카드 혜택 조회 및 비교
card_info_template = ChatPromptTemplate.from_template("""
너는 한국어로 신용카드 혜택을 친절하게 설명하는 전문가야.

- 처음에는 사용자의 질문에 관련된 신용카드 목록을 보여줘. 각 카드별로 이름과 주요 혜택을 간략하게 요약해줘.
- 사용자가 원하는 카드 개수를 말하지 않았다면 최대한 다양한 옵션을 보여줘.
- 카드명은 context에 등장한 실제 카드명만 사용하고, 간단하고 명확한 표현으로 요약해줘.
- 리스트가 끝나면, 그 안에서 **가장 혜택이 좋은 카드 1~2개만 추천**해
- 마지막에는 혜택 비교 및 추천해줘 추천한 기준도 간단히 정리해줘

# 추천 신용카드 목록:
{context}

# 이전 대화:
{chat_history}

# 질문:
{question}

# 답변:
""")
compare_chain = {
        "context": reranker | RunnableLambda(format_docs), # 리랭커: retriever0 -> reranker
        "chat_history": RunnableLambda(get_chat_history),
        "question": RunnablePassthrough(),
} | card_info_template | llm

# 3번 응답: 사용자 맞춤 추천 (역질문)
def ask_user_profile():
    answers = {}
    questions = [
        "그렇다면 카드를 추천드리기에 앞서 몇가지 질문을 드리겠습니다. 나이가 어떻게 되십니까?",
        "성별은 무엇이신가요?",
        "직업은 무엇이신가요?",
        "마지막으로 관심사나 취미가 있다면 알려주세요. 영화/책/드라이브/애완동물 등 무엇이든 좋습니다."
    ]
    for q in questions:
        answers[q] = input(f"{q} ")
    profile = "\n".join([f"{k} {v}" for k, v in answers.items()])
    prompt = f"""

너는 한국어로 신용카드 혜택을 친절하게 설명하는 전문가야.

-아래 사용자 프로필을 참고해서 가장 적절한 신용카드를 3개를 아래 형식으로 답변해줘.
답변은 다음 형식을 따라줘:

1. 카드 이름
- 주요 혜택 요약
- 그 외 혜택 요약
2. 카드 이름
- 주요 혜택 요약
- 그 외 혜택 요약
...

사용자 프로필:
{profile}

#답변:
"""
    return llm.invoke(prompt)

# 4번 응답: 무관한 질문
irrelevant_response = "그 질문은 카드나 혜택이랑은 관련이 없습니다."

# template for web search result.
template_for_web = """
너는 사용자의 질문에 대한 답변을 잘 설명해주는 어시스턴트야.
반드시 한국어로만 답변하고, 영어는 절대 사용하지 마.
아래 검색 결과(context)에서 사용자 질문에 대한 답변을 찾아서 친절하고 구조화된 형식으로 답변해줘.
검색 결과(context)에서 질문에 대한 답변을 찾지 못했으면, 답변을 찾지 못했다고 말해줘.

답변은 다음 형식을 따라줘:
웹에서 검색한 결과는 다음과 같습니다.
...

#질문:
{question}

#검색 결과:
{context}

#답변:
"""

prompt_for_web = ChatPromptTemplate.from_template(template_for_web)

web_chain = {
        "context": RunnablePassthrough(), # 리랭커: retriever0 -> reranker
        #"chat_history": RunnableLambda(get_chat_history),
        "question": RunnablePassthrough(),
} | prompt_for_web | llm

def enhance_query(query):
    if len(query.strip()) < 2:
        return query + " 의미"  # 단어가 너무 짧으면 "의미"를 추가
    return query



# === 실행부 ===
if __name__ == "__main__":
    print("\n💳 신용카드 GPT 차트및 (최종 버전)")
    while True:
        user_input = input("\n질문 입력 ('q' 또는 '종료' 입력 시 종료): ").strip()

        if user_input.lower() in ["q", "종료"]:
            print("\n차트및을 종료합니다. 감사합니다!")
            break

        try:
            category = classifier_chain.invoke({"question": user_input}).strip()

            if category == "1":
                result = recommend_chain.invoke(user_input)

            elif category == "2":
                result = compare_chain.invoke(user_input)

            elif category == "3":
                result = ask_user_profile()

            else:
                print("4번")
                search_query = user_input
                search_query = enhance_query(search_query)
                web_search = TavilySearchResults(max_results=2)
                search_results = web_search.invoke(search_query)
                search_result = ''
                for result in search_results:
                    search_result += result['content'] + '\n'
                result = web_chain.invoke(
                    {
                        "question": user_input,
                        "context": search_result,
                    }
                )
            print("\n답변:", result, "\n")
            memory.save_context({"question": user_input}, {"output": result})

        except Exception as e:
            print("오류 발생:", str(e))
            