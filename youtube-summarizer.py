# -*- coding: utf-8 -*-
"""youtube-summarizer

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bUyLCUoZN5edvEWONVaF3nUvG14z7yP7
"""

#!pip install ffmpeg-python streamlit yt-dlp openai-whisper openai langchain langchain_community

import ffmpeg
import streamlit as st
import yt_dlp
import whisper
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os

# Streamlit 앱 제목 설정
st.set_page_config(page_title="YouTube 영상 요약 서비스", page_icon="📺")

# 제목과 부제목
st.title("YouTube 영상 요약 서비스")
st.subheader("AI를 활용한 영상 내용 요약 보고서")

# OpenAI API 키 입력
openai_api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")
os.environ["OPENAI_API_KEY"] = openai_api_key

# YouTube URL 입력
youtube_url = st.text_input("YouTube 영상 URL을 입력하세요:")

if st.button("영상 요약하기"):
    if not openai_api_key:
        st.error("OpenAI API 키를 입력해주세요.")
    elif not youtube_url:
        st.error("YouTube 영상 URL을 입력해주세요.")
    else:
        try:
            # 진행 상태 표시
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 1. YouTube 영상 다운로드
            status_text.text("YouTube 영상 다운로드 중...")
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
                ydl.download([youtube_url])
            progress_bar.progress(25)

            # 2. 음성 인식 (Whisper)
            status_text.text("음성을 텍스트로 변환 중...")
            model = whisper.load_model("base")
            result = model.transcribe("audio.mp3")
            transcription = result["text"]
            progress_bar.progress(50)

            # 3. 텍스트 요약 (GPT-4)
            status_text.text("텍스트 요약 중...")
            chat = ChatOpenAI(model_name="gpt-4", temperature=0)
            messages = [
                SystemMessage(content="You are a helpful assistant that summarizes YouTube video transcripts."),
                HumanMessage(content=f"Please summarize the following transcript in a detailed report format:\n\n{transcription}")
            ]
            summary = chat(messages).content
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
st.markdown("**주의사항:**")
st.markdown("- 이 서비스는 OpenAI API를 사용하므로 API 사용량에 따라 비용이 발생할 수 있습니다.")
st.markdown("- 영상의 길이와 복잡도에 따라 처리 시간이 달라질 수 있습니다.")
st.markdown("- 저작권 보호를 위해 개인적인 용도로만 사용해주세요.")