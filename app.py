import streamlit as st

from rag_app import GEMINI_MODEL, ask


st.set_page_config(
    page_title="HR RAG Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 HR RAG Assistant")
st.caption(f"LangGraph + ChromaDB + Gemini ({GEMINI_MODEL})")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask an HR, billing, or general question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask(question)
                answer = result["answer"]
                category = result.get("category", "unknown")
                st.caption(f"Route: {category}")
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as exc:
                st.error(f"Application error: {exc}")
                st.info(
                    "Make sure GEMINI_API_KEY is configured in Replit Secrets."
                )

with st.sidebar:
    st.header("Demo questions")
    st.write("Try these:")
    st.code("How many days can I work from home?")
    st.code("How many annual leave days do employees get?")
    st.code("Can I get a refund for my annual subscription?")
    st.code("What is Kubernetes?")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
