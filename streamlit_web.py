
import streamlit as st
from main import (
    classifier_chain,
    recommend_chain,
    compare_chain,
    web_chain,
    enhance_query,
    memory,
    llm
)
from langchain_community.tools import TavilySearchResults


def stringify_response(response):
    if hasattr(response, "content"):
        return response.content
    if isinstance(response, dict) and "output" in response:
        return str(response["output"])

    def flatten(l):
        for item in l:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    if isinstance(response, list):
        flat = list(flatten(response))
        return "\n".join(map(str, flat))

    if isinstance(response, str):
        return response

    return str(response)


st.set_page_config(page_title="신용카드 혜택 챗봇", layout="centered")
st.title("💳 신용카드 혜택 챗봇")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "show_profile_form" not in st.session_state:
    st.session_state.show_profile_form = False

user_input = st.chat_input("카드에 대해 궁금한 걸 입력해보세요!")

if user_input:
    with st.spinner("답변 생성 중..."):
        try:
            category = classifier_chain.invoke({"question": user_input}).strip()

            if category == "1":
                response = recommend_chain.invoke(user_input)
                st.session_state.show_profile_form = False

            elif category == "2":
                response = compare_chain.invoke(user_input)
                st.session_state.show_profile_form = False

            elif category == "3":
                response = "맞춤형 추천을 위해 아래 입력폼을 작성해주세요!"
                st.session_state.show_profile_form = True

            else:
                search_query = enhance_query(user_input)
                web_search = TavilySearchResults(max_results=2)
                search_results = web_search.invoke(search_query)
                search_context = "\n".join([r["content"] for r in search_results])
                response = web_chain.invoke({
                    "question": user_input,
                    "context": search_context
                })
                st.session_state.show_profile_form = False

            response = stringify_response(response)
            st.session_state.chat_history.append((user_input, response))
            memory.save_context({"question": user_input}, {"output": response})

        except Exception as e:
            st.session_state.chat_history.append((user_input, f"❗ 오류 발생: {str(e)}"))

# 채팅 기록 출력
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# 사용자 맞춤 추천 입력 폼
if st.session_state.show_profile_form:
    st.markdown("---")
    st.subheader("🧑‍💼 사용자 맞춤 카드 추천")

    with st.form("user_profile_form"):
        st.markdown("**사용자 맞춤 추천을 위해 아래 내용을 입력해주세요.**")
        age = st.text_input("나이는 어떻게 되시나요?")
        gender = st.selectbox("성별을 선택해주세요", ["선택 안 함", "남성", "여성"])
        job = st.text_input("직업을 알려주세요")
        hobby = st.text_input("취미나 관심사 (예: 영화, 커피, 쇼핑 등)")

        submitted = st.form_submit_button("추천 카드 보기")

    if submitted:
        profile = f"나이: {age}\n성별: {gender}\n직업: {job}\n관심사: {hobby}"
        prompt = f"""
너는 신용카드 혜택을 잘 설명해주는 어시스턴트야.
반드시 한국어로만 답변하고, 영어나 다른 언어는 절대 사용하지 마.

아래 사용자 프로필을 참고해서 가장 적절한 신용카드 3개를 추천해줘.
카드 이름과 주요 혜택만 간단히 정리해줘. 한줄평은 넣지 마.

사용자 프로필:
{profile}

#추천 카드:
"""
        with st.spinner("맞춤형 카드 추천 중..."):
            try:
                result = llm.invoke(prompt)
                st.chat_message("assistant").markdown(stringify_response(result))
            except Exception as e:
                st.error(f"❗ 오류 발생: {str(e)}")
