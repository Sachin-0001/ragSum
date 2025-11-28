import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
st.set_page_config(page_title="ChatCut")

# Navigation buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("RAG Chatbot"):
        st.switch_page("main.py")

with col2:
    st.button("Text Summarizer", type="primary")
        # st.switch_page("pages/summ.py") 

st.header("ChatCut")
model_name = "Sachin-0001/dialogsum-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

st.title("Summarize Text")

text = st.text_area("Enter Text:",height = 250)

if st.button("Summarize"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    output = model.generate(**inputs, max_length=60)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write("### Summary:")
    st.write(summary)