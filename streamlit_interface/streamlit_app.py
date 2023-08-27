import sys
# append parent directory to sys.path
sys.path.append("..")
from src.summarizer import llama_summarizer
import streamlit as st

# title of the app
st.title("OLlama for text summarization")

with st.sidebar:
    retriever_opt = st.selectbox(
        "Retriever", ("default", "SVM", "MultiQuery")
    )
    device_opt = st.selectbox("Device", ("mps", "cpu", "cuda"))
    model_opt = st.selectbox("Llama", ("summarizev2", "summarize"))
    embedding_opt = st.selectbox("Embedding model", ("large", "small"))
    # line breaks
    st.write("\n")
    st.write("\n")

    col1, col2 = st.columns([2, 1])

    # write in very small text
    with col1:
        st.markdown(
            "<p style='font-size:12px'>The summarizer is based on the OLLAMA framework.</p>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f'<a href="https://your-url.com" target="_blank" style="font-size:12px;">Link to website</a>',
            unsafe_allow_html=True,
        )

    # line breaks
    st.write("\n")
    st.write("\n")

    col3, col4 = st.columns([2, 1])

    with col3:
        st.markdown(
            "<p style='font-size:12px'>Click 'clear' to reset caches.</p>",
            unsafe_allow_html=True,
        )

    with col4:
        # Button to rerun the app (start from fresh)
        if st.button("Clear", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.experimental_rerun()
            url = ""
            question = ""
            summary = ""
            box_height = 0

# with col1:
url = st.text_input(
    "Enter URL to the text you want to summarize",
    placeholder="https://andersen.sdu.dk/vaerk/hersholt/TheUglyDuckling_e.html",
)
question = st.text_input(
    "Enter a question to ask the model", placeholder="Summarize this text:"
)


enable_button = bool(url and question)


# define function to do summarization, to re initialize the model.
def run_summarizer(url, question, retriever, device, model, embedding_model):
    summarizer = llama_summarizer(
        url=url,
        question=question,
        retriever=retriever,
        device=device,
        model=model,
        embedding_model=embedding_model,
    )
    summarizer.generate()
    return summarizer.answ["result"].strip()


st.text("")


if st.button(
    "Summarize", use_container_width=True, disabled=not enable_button
):
    # button trigger summarization
    with st.spinner("Summarizing..."):
        summary = run_summarizer(
            url,
            question,
            retriever_opt,
            device_opt,
            model_opt,
            embedding_opt,
        )

    box_height = int(len(summary) * 0.35)
    st.text_area(
        "Summary", value=summary, height=box_height, max_chars=None, key=None
    )
