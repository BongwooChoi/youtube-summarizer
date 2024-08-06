#!pip install ffmpeg-python streamlit yt-dlp openai-whisper openai langchain langchain_community youtube-transcript-api

import streamlit as st
import yt_dlp
import whisper
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
from youtube_transcript_api import YouTubeTranscriptApi

# Streamlit 앱 제목 설정
st.set_page_config(page_title="YouTube 영상 요약 서비스", page_icon="📺")
st.title("📺 YouTube 영상 요약봇")
st.subheader("바쁜 현대인, 당신의 시간을 절약해 드립니다.")

# OpenAI API 키 가져오기
openai_api_key = st.secrets["openai_api_key"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# YouTube URL 입력
youtube_url = st.text_input("YouTube 영상 URL을 입력하세요:")

def get_video_id(url):
    """YouTube URL에서 비디오 ID를 추출합니다."""
    if "youtu.be" in url:
        return url.split("/")[-1]
    elif "youtube.com" in url:
        return url.split("v=")[1].split("&")[0]
    else:
        raise ValueError("올바른 YouTube URL이 아닙니다.")

def get_transcript(video_id):
    """YouTube 비디오의 자막을 가져옵니다."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        st.warning(f"자막을 가져오는 데 실패했습니다. 음성 인식을 시도합니다. 오류: {str(e)}")
        return None

def transcribe_audio(url):
    """YouTube 비디오의 오디오를 다운로드하고 Whisper로 음성을 인식합니다."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'audio.%(ext)s'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    model = whisper.load_model("base")
    result = model.transcribe("audio.mp3")
    return result["text"]

def summarize_text(text):
    """GPT-4o-mini를 사용하여 텍스트를 요약합니다."""
    chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    messages = [
        SystemMessage(content="당신은 나 대신 YouTube를 시청하고 내용을 정리해서 알려주는 Assistant입니다."),
        HumanMessage(content=f"아래 내용을 가독성 있는 한 페이지의 보고서 형태로 요약하세요. 최종결과는 한국어로 나와야 합니다.:\n\n{text}")
    ]
    return chat(messages).content

if st.button("영상 요약하기"):
    if not openai_api_key:
        st.error("OpenAI API 키를 입력해주세요.")
    elif not youtube_url:
        st.error("YouTube 영상 URL을 입력해주세요.")
    else:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            video_id = get_video_id(youtube_url)
            
            # 1. 자막 가져오기 시도
            status_text.text("자막 가져오는 중...")
            transcript = get_transcript(video_id)
            progress_bar.progress(25)
            
            # 2. 자막이 없으면 음성 인식 수행
            if transcript is None:
                status_text.text("음성을 텍스트로 변환 중...")
                transcript = transcribe_audio(youtube_url)
            progress_bar.progress(50)
            
            # 3. 텍스트 요약 (GPT-4)
            status_text.text("텍스트 요약 중...")
            summary = summarize_text(transcript)
            progress_bar.progress(75)
            
            # 4. 결과 표시
            status_text.text("요약 완료!")
            st.subheader("영상 요약 결과")
            st.write(summary)
            progress_bar.progress(100)
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")

# 주의사항 및 안내
st.markdown("---")
st.markdown("**안내사항:**")
st.markdown("- 이 서비스는 YouTube 자막 또는 OpenAI의 Whisper(음성 인식)와 GPT-4(텍스트 요약)를 사용합니다.")
st.markdown("- 영상의 길이와 복잡도에 따라 처리 시간이 달라질 수 있습니다.")
st.markdown("- 저작권 보호를 위해 개인적인 용도로만 사용해주세요.")
