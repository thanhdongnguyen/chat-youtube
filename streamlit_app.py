import io
import os
from contextlib import suppress
from urllib.parse import urlparse, parse_qs

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

import webvtt
from datetime import datetime, time

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter, WebVTTFormatter

import streamlit as st

# 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
# os.environ["OPENAI_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]  = "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"

# APIKEY = os.environ["OPENAI_API_KEY"]
# videoId = "MSEoOkg8oPY"

@st.cache_data()
def getSub(video_id):

    path = f"{video_id}.txt"
    trans = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "vi"], preserve_formatting=True)
    formater = TextFormatter()
    text_format = formater.format_transcript(trans)


    with open(path, 'w', encoding='utf-8') as file:
        file.write(text_format)

    loader = TextLoader(path)
    docs = loader.load()

    os.remove(path)

    print("Loaded docs: DONE")
    return docs

@st.cache_data()
def getWtt(video_id):

    path = f"{video_id}.vtt"
    trans = YouTubeTranscriptApi.get_transcript(video_id, languages=["vi", "en"], preserve_formatting=True)
    formater = WebVTTFormatter()
    text_format = formater.format_transcript(trans)

    with open(path, 'w', encoding='utf-8') as file:
        file.write(text_format)


    return path


def time_to_webvtt_timestamp(t: time):
    """Convert a datetime.time object to a WebVTT timestamp string."""
    return f"{t.strftime('%H:%M:%S')}.000"


def string_to_time(s: str):
    """Convert a string to a datetime.time object."""
    return datetime.strptime(s, "%H:%M:%S.%f").time()

def vtt_string_to_dataframe(vtt_string: str) -> pd.DataFrame:
    time_epsilon = pd.Timedelta("00:00:00.1")

    buffer = io.StringIO(vtt_string)

    vtt = webvtt.read_buffer(buffer=buffer)

    df = pd.DataFrame(
        [
            [
                pd.to_datetime(v.start),
                pd.to_datetime(v.end),
                v.text.splitlines()[-1],
            ]
            for v in vtt
        ],
        columns=["start", "end", "text"],
    )
    df = df.where(df.end - df.start > time_epsilon).dropna()
    df["start"] = df["start"].apply(time_to_webvtt_timestamp)
    df["end"] = df["end"].apply(time_to_webvtt_timestamp)
    df["start"] = df["start"].apply(string_to_time)
    df["end"] = df["end"].apply(string_to_time)
    return df

def init_db(api_key: str, video_id: str):
    embeds = OpenAIEmbeddings(api_key=api_key)
    textSplit = RecursiveCharacterTextSplitter()
    docs = textSplit.split_documents(getSub(video_id))
    vector = FAISS.from_documents(docs, embeds)
    print("Init DB: DONE")
    return vector.as_retriever()


@st.cache_data()
def get_yt_id(url, ignore_playlist=False):
    # Examples:
    # - http://youtu.be/SA2iWivDJiE
    # - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    # - http://www.youtube.com/embed/SA2iWivDJiE
    # - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    query = urlparse(url)
    if query.hostname == 'youtu.be': return query.path[1:]
    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if not ignore_playlist:
        # use case: get playlist id not current video in playlist
            with suppress(KeyError):
                return parse_qs(query.query)['list'][0]
        if query.path == '/watch': return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/watch/': return query.path.split('/')[2]
        if query.path[:7] == '/embed/': return query.path.split('/')[2]
        if query.path[:3] == '/v/': return query.path.split('/')[2]
   # returns None for invalid YouTube url

@st.cache_data()
def genAI(api_key: str, question: str, video_id: str):


    # template = ChatPromptTemplate.from_template("""
    #     Nhiệm vụ của bạn là trả lời và phiên dịch những nội dung dựa trên kịch bản đã cung cấp:
    #     <context>
    #     {context}
    #     </context>
    #
    #     Question: {input}
    # """)
    template = ChatPromptTemplate.from_messages([
        ("system", """
              Nhiệm vụ của bạn là trả lời và phiên dịch những nội dung dựa trên kịch bản đã cung cấp. Và hãy trả lời theo ngôn ngữ mà người dùng đã hỏi:
                <context>
                 {context}
                </context>
        """),
        MessagesPlaceholder(variable_name="history_data"),
        ("user", "Question: {input}"),
    ])
    llm = ChatOpenAI(model="gpt-4-turbo", api_key=api_key)

    doc_chain = create_stuff_documents_chain(llm, template)
    re_chain = create_retrieval_chain(init_db(api_key, video_id), doc_chain)

    c = re_chain

    res = c.invoke({
        "history_data": [],
        "input": question,
    })
    return res["answer"]


st.header("Chat with Youtube")
st.caption("Enter your question and link video Youtube to chat with the AI")

with st.sidebar:
    openai_key = st.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    link_youtube = st.text_input("Link Youtube", value="https://www.youtube.com/watch?v=MSEoOkg8oPY")

    if link_youtube:
        st.video(link_youtube)

        with st.status("Loading subtitles..."):
            pathWtt = getWtt(get_yt_id(link_youtube))
        if pathWtt:
            df = vtt_string_to_dataframe(open(pathWtt).read())
            "Subtitles in video:"
            st.dataframe(df)

            if os.path.exists(pathWtt):
                os.remove(pathWtt)


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "Bot", "message": "Hello, I'm AI. I can help you answer questions about the video"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["message"])


if prompt := st.chat_input():
    if not openai_key:
        st.info("Please enter OpenAI API Key")
        st.stop()
    if not link_youtube:
        st.info("Please enter link Youtube")
        st.stop()

    videoId = get_yt_id(link_youtube)

    print(videoId, openai_key, link_youtube)
    if not videoId:
        st.error("Invalid Youtube link")
        st.stop()

    st.session_state["messages"].append({"role": "User", "message": prompt})

    st.chat_message("User").write(prompt)

    with st.status("Generating AI response..."):
        res = genAI(openai_key, prompt, videoId)

    st.session_state["messages"].append({"role": "Bot", "message": res})

    st.chat_message("Bot").write(res)

