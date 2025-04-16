import streamlit as st
from scripts.chatbot import ask_rag
import jiter
print(dir(jiter))  # List all functions and attributes in the jiter package


st.set_page_config(page_title="학습 멘토 GPT", page_icon="🧠")
st.title("📚 학습자 고민 상담 챗봇")

user_input = st.text_input("고민을 입력해 주세요 (예: 집중이 잘 안 돼요)", "")

if user_input:
    with st.spinner("답변 생성 중..."):
        answer = ask_rag(user_input)
        print(f"답변: {answer}")
        st.markdown("🧠 **답변**")
        st.write(answer)
