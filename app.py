
import streamlit as st
import pandas as pd
import random
import json
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from faker import Faker
from google import genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import tempfile
import os

st.set_page_config(
    page_title="Chatbot INAPROC - AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    :root {
        --merah-inaproc: #B22222;
        --merah-muda: #FFE4E1;
        --merah-tua: #8B0000;
    }
    .stButton > button {
        background-color: var(--merah-inaproc);
        color: white;
        border: none;
        border-radius: 8px;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: var(--merah-tua);
    }
    .stChatInput input {
        border: 2px solid var(--merah-inaproc);
        border-radius: 20px;
    }
    [data-testid="stSidebar"] {
        background-color: #FFF8F5;
        border-right: 3px solid var(--merah-inaproc);
    }
    .user-profile-card {
        background-color: #FFE4E1;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .role-badge {
        background-color: #B22222;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

try:
    API_KEY = st.secrets["google_api_key"]
    client = genai.Client(api_key=API_KEY)
    st.success("✅ Koneksi ke Gemini API berhasil!")
except Exception as e:
    st.error(f"❌ Gagal mengambil API Key: {e}")
    st.info("""
    **Cara Setting Secrets di Streamlit Cloud:**
    1. Buka dashboard aplikasi di share.streamlit.io
    2. Klik Settings (ikon gerigi)
    3. Scroll ke Secrets
    4. Tambahkan:
    
    ```toml
    google_api_key = "API_KEY_KAMU"