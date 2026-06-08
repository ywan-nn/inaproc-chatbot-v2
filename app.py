# app.py - Chatbot INAPROC dengan User Profiling & RAG
# API key dibaca dari st.secrets, JANGAN hardcode!

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

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(
    page_title="Chatbot INAPROC - AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# ==================== PALETTE WARNA MERAH ====================
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

# ==================== AMBIL API KEY DARI SECRETS ====================
try:
    API_KEY = st.secrets["google_api_key"]
    client = genai.Client(api_key=API_KEY)
    st.success("✅ Koneksi ke Gemini API berhasil!")
except Exception as e:
    st.error(f"❌ Gagal mengambil API Key: {e}")
    st.info(
        "**Cara Setting Secrets di Streamlit Cloud:**\n\n"
        "1. Buka dashboard aplikasi di share.streamlit.io\n"
        "2. Klik Settings (ikon gerigi)\n"
        "3. Scroll ke Secrets\n"
        "4. Tambahkan:\n\n"
        "   google_api_key = 'API_KEY_KAMU'\n\n"
        "5. Klik Save"
    )
    st.stop()

# ==================== USER PROFILING ====================
USER_CREDENTIALS = {
    "ppk_budi": {"password": "12345", "nama": "Budi Santoso", "role": "PPK", "unit": "Dinas Pendidikan", "preferences": {"prioritas": ["harga", "waktu_pengiriman"]}},
    "staff_siti": {"password": "12345", "nama": "Siti Aminah", "role": "Staff Pengadaan", "unit": "Dinas Kesehatan", "preferences": {"prioritas": ["kualitas", "garansi"]}},
    "vendor_ahmad": {"password": "12345", "nama": "Ahmad Wijaya", "role": "Vendor", "unit": "PT Maju Jaya", "preferences": {"prioritas": ["harga_kompetitif"]}},
    "admin": {"password": "admin123", "nama": "Admin INAPROC", "role": "Administrator", "unit": "LKPP", "preferences": {"prioritas": ["kepatuhan"]}}
}

def check_login(username: str, password: str) -> bool:
    if username in USER_CREDENTIALS:
        return USER_CREDENTIALS[username]["password"] == password
    return False

def get_user_profile(username: str) -> dict:
    return USER_CREDENTIALS.get(username)

# ==================== TAMPILAN LOGIN ====================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.user_profile = None
    st.session_state.messages = []
    st.session_state.active_po = None
    st.session_state.vectorstore = None

if not st.session_state.logged_in:
    st.title("🔐 Login Chatbot INAPROC")
    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", type="primary"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.current_user = username
                st.session_state.user_profile = get_user_profile(username)
                st.session_state.messages = []
                st.rerun()
            else:
                st.error("❌ Username atau password salah!")
    with col2:
        st.markdown("**Akun Demo:**\n- ppk_budi / 12345\n- staff_siti / 12345\n- vendor_ahmad / 12345\n- admin / admin123")
    st.stop()

# ==================== RAG ====================
@st.cache_resource
def build_knowledge_base():
    with st.spinner("Memuat pengetahuan dari Pusat Bantuan INAPROC..."):
        urls = [
            "https://bantuan.inaproc.id/hc/id-id/sections/9000771875471-Pertanyaan-yang-Sering-Tanyakan-Seputar-Manajemen-Akun-SPSE",
            "https://bantuan.inaproc.id/hc/id-id/sections/9031756154767-Panduan-Penyedia"
        ]
        all_docs = []
        fallback = "INAPROC adalah platform pengadaan digital LKPP. Pusat Bantuan: layanan@lkpp.go.id, WhatsApp 08111557709, Call Center 144."
        all_docs.append(Document(page_content=fallback, metadata={"source": "fallback"}))
        
        for url in urls:
            try:
                r = requests.get(url, timeout=30)
                soup = BeautifulSoup(r.content, 'html.parser')
                content = soup.find('div', class_='article-body')
                if content:
                    text = content.get_text(separator='\n', strip=True)
                    all_docs.append(Document(page_content=text[:3000], metadata={"source": url}))
            except:
                pass
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return FAISS.from_documents(chunks, embeddings)

if st.session_state.vectorstore is None:
    st.session_state.vectorstore = build_knowledge_base()

# ==================== DATA DUMMY ====================
@st.cache_data
def generate_orders():
    fake = Faker('id_ID')
    orders = {}
    for i in range(1, 31):
        po = f"PO-2025-{str(10000 + i)[1:]}"
        orders[po] = {
            "status": random.choice(["SHIPPED", "DELAYED", "DELIVERED"]),
            "kurir": random.choice(["JNE", "SiCepat"]),
            "resi": str(fake.random_number(digits=12)),
            "lokasi": fake.city(),
            "produk": random.choice(["Laptop ASUS", "AC Panasonic", "Proyektor Epson"]),
            "vendor": random.choice(["PT Maju Jaya", "CV Sukses Abadi"])
        }
    return orders

if "orders_db" not in st.session_state:
    st.session_state.orders_db = generate_orders()

# ==================== FUNGSI CHATBOT ====================
def check_order_status(po_number: str) -> dict:
    if po_number in st.session_state.orders_db:
        return {"found": True, **st.session_state.orders_db[po_number]}
    return {"found": False}

def search_kb(query: str) -> str:
    try:
        docs = st.session_state.vectorstore.similarity_search(query, k=2)
        return "\n".join([d.page_content for d in docs])
    except:
        return ""

def get_bot_response(user_input: str) -> str:
    po_match = re.search(r'PO[-_]?\d{4,}[-_]?\d{3,}', user_input.upper())
    order_info = ""
    if po_match:
        po = po_match.group(0)
        st.session_state.active_po = po
        status = check_order_status(po)
        if status["found"]:
            order_info = f"Status: {status['status']}, Produk: {status['produk']}, Vendor: {status['vendor']}"
    
    kb_context = search_kb(user_input)
    profile = st.session_state.user_profile
    role = profile.get("role", "User")
    
    history = ""
    if st.session_state.messages:
        for msg in st.session_state.messages[-4:]:
            history += f"{msg['role']}: {msg['content'][:100]}...\n"
    
    prompt = f"""
    Kamu asisten INAPROC. Role user: {role}
    
    Info Pusat Bantuan: {kb_context[:1500]}
    {f"Info Pesanan: {order_info}" if order_info else ""}
    
    Percakapan sebelumnya:
    {history}
    
    User: {user_input}
    
    Jawab dengan bahasa Indonesia yang ramah dan sesuai role user. Jangan minta PO lagi jika sudah ada.
    """
    
    try:
        r = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return r.text
    except Exception as e:
        return f"Error: {str(e)}"

# ==================== UI ====================
profile = st.session_state.user_profile

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.title("🤖 Chatbot INAPROC")
with col2:
    st.markdown(f"""
    <div class="user-profile-card">
        <strong>👤 {profile.get('nama', 'User')}</strong><br>
        <span class="role-badge">{profile.get('role', 'User')}</span>
    </div>
    """, unsafe_allow_html=True)
with col3:
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Tanyakan status pesanan atau pertanyaan seputar INAPROC..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Memproses..."):
            resp = get_bot_response(prompt)
            st.markdown(resp)
    st.session_state.messages.append({"role": "assistant", "content": resp})

with st.sidebar:
    st.markdown(f"### 👋 {profile.get('nama', 'User')}")
    if st.session_state.active_po:
        st.info(f"PO Aktif: `{st.session_state.active_po}`")
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.session_state.active_po = None
        st.rerun()
