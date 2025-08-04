import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Text Loader
from langchain_community.document_loaders import TextLoader
# PDF Loader
from langchain_community.document_loaders import PyPDFLoader
# Audio Loader
from langchain_community.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain_core.documents import Document
#Video Loader
from youtube_transcript_api import YouTubeTranscriptApi
## Web based loader
from langchain_community.document_loaders import WebBaseLoader
import bs4
#Research Paper Loader
from langchain_community.document_loaders import ArxivLoader
import re
#wikipedia
from langchain_community.document_loaders import WikipediaLoader

import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith Tracking
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"]="SmartGist"

#Title of the app
st.markdown(
    """
    <div>
        <span style='color: #1E90FF; font-size: 47px; font-weight: bold;'>🐬SmartGist </span>
        <span style='color: #25E817; font-size: 31px;'> - AI Powered Multi Summarizer</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.set_page_config(page_title="SmartGist", page_icon="🐬")

#Model Selection
options = ["gemma-3n-e2b-it", "gemma-3n-e4b-it",]
Model = st.sidebar._selectbox("Choose Model", options)

#LLM model
llm = GoogleGenerativeAI(model=Model, google_api_key=os.getenv("GOOGLE_API_KEY"))

loader = Document(page_content="", metadata={})
# docs = list()

inputs = st.chat_input(
        "what do you want to summarize?",
        accept_file="multiple",
        file_type=["txt", "pdf", "mp3", "wav", "ogg", "m4a"],
        disabled=False
    )

file_type = inputs["files"][0].type if inputs and inputs.get("files") else None
file_name = inputs["files"][0].name if inputs and inputs.get("files") else None

try:
    if inputs or inputs.get("files"):
        for uploaded_file in inputs["files"]:
            file_name = uploaded_file.name
            file_type = uploaded_file.type
            temp = f"./{file_name}"

            # st.write(f"📁 Saving: {temp}")
            with open(temp, "wb") as f:
                f.write(uploaded_file.getvalue())
except Exception as e:
    pass

if file_type == "text/plain":
    loader=TextLoader(temp)
    docs=loader.load_and_split()

elif file_type == "application/pdf":
    loader = PyPDFLoader(temp)
    docs=loader.load_and_split()

elif file_type in ["audio/mpeg", "audio/wav", "audio/ogg", "audio/x-m4a"]:
    loader = AssemblyAIAudioTranscriptLoader(file_path=temp, api_key=os.getenv("ASSEMBLYAI_API_KEY"))
    docs=loader.load_and_split()

elif inputs and "www.youtube.com" in inputs.text:
    # try:
    video_id = inputs.text.split("=")[1]
    # st.write(video_id)
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)
    transcript = ""
    for snippet in fetched_transcript:
        print(snippet.text)
        transcript += " " + snippet.text
    
    script = Document(page_content=transcript, metadata={"source": "youtube_transcript"})
    docs = [script]

elif inputs and ("https://" in inputs.text or "http://" in inputs.text or "www." in inputs.text):
    loader=WebBaseLoader(web_paths=(inputs.text,))
    docs=loader.load()

elif inputs and ("https://arxiv.org/" in inputs.text or bool(re.fullmatch(r'\d{4}\.\d{5}', inputs.text))):
    docs = ArxivLoader(query="1706.03762", load_max_docs=2).load()

elif inputs and bool(re.fullmatch(r'[A-Za-z0-9\W_]+', inputs.text)):
            docs = WikipediaLoader(query=inputs.text, load_max_docs=2).load()

try:
    with st.spinner("summarizing..."):
        final_docs=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100).split_documents(docs)

        chain=load_summarize_chain(
            llm=llm,
            chain_type="refine",
            verbose=True
        )
        output_summary=chain.run(final_docs[:5])
        st.markdown("### Summary:")
        st.write(output_summary)

except Exception as e:
    pass


