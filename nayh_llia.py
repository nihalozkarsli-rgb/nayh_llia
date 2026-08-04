"""
nayh_llia — Kişisel Yapay Zeka Asistanı (Groq — ücretsiz API ile)
====================================================================

KURULUM:
    pip install streamlit openai streamlit-mic-recorder gTTS

    (Groq, OpenAI ile uyumlu bir API sunuyor, bu yüzden "openai" kütüphanesini
    kullanıyoruz ama istekler Groq'un sunucularına gidiyor — OpenAI'ye para
    ödemiyoruz, hesap da açmıyoruz.)

API ANAHTARI ALMA (ÜCRETSİZ, KREDİ KARTI İSTEMİYOR):
    1) https://console.groq.com adresine git
    2) Google hesabınla giriş yap
    3) Sol menüden "API Keys" -> "Create API Key"
    4) Çıkan anahtarı kopyala (gsk_ ile başlar)

ÇALIŞTIRMA:
    py -m streamlit run streamlit_app.py
"""

import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

# ------------------------------------------------------------------
# 1) SAYFA AYARLARI
# ------------------------------------------------------------------
st.set_page_config(
    page_title="nayh_llia",
    page_icon="🌙",
    layout="centered",
)

# ------------------------------------------------------------------
# 2) MOR / GECE TEMASI (CSS)
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1440 50%, #24123f 100%);
        color: #f0eaff;
    }
    section[data-testid="stSidebar"] {
        background-color: #150f30;
    }
    .stChatMessage {
        border-radius: 14px;
    }
    h1, h2, h3 {
        color: #c9a8ff !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: #1f1748;
        color: #f0eaff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 3) BAŞLIK / MARKA ALANI
# ------------------------------------------------------------------
col1, col2 = st.columns([1, 4])
with col1:
    try:
        st.image("logo.jpg", width=80)
    except Exception:
        st.write("🌙")
with col2:
    st.markdown("## nayh_llia")
    st.caption("Kişisel yapay zeka asistanın")

st.divider()

# ------------------------------------------------------------------
# 4) API ANAHTARI
# ------------------------------------------------------------------
# console.groq.com'dan aldığın anahtarı buraya yapıştır (gsk_ ile başlar)
API_KEY = "BURAYA_KENDI_GROQ_API_ANAHTARINI_YAPISTIR"
# .strip() -> kopyalarken gelebilecek görünmeyen boşluk/satır sonlarını temizler
API_KEY = API_KEY.strip()

# Geçici kontrol: anahtarın doğru yapıştırıldığını doğrulamak için
# (uzunluk ve ilk/son karakterleri gösterir, tam anahtarı göstermez)
if API_KEY == "BURAYA_KENDI_GROQ_API_ANAHTARINI_YAPISTIR":
    st.error("API_KEY satırı hâlâ placeholder — anahtarını yapıştırmayı unutmuşsun.")
    st.stop()
else:
    st.caption(f"🔑 Anahtar yüklendi: {API_KEY[:6]}...{API_KEY[-4:]} ({len(API_KEY)} karakter)")

if "client" not in st.session_state:
    try:
        st.session_state.client = OpenAI(
            api_key=API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
    except Exception as e:
        st.error(f"API bağlantısı kurulamadı: {e}")
        st.stop()

client = st.session_state.client

# ------------------------------------------------------------------
# 5) SİSTEM TALİMATI (asistanın kişiliği / uzmanlığı)
# ------------------------------------------------------------------
SISTEM_TALIMATI = """
Sen nayh_llia adında bir yapay zeka asistanısın. Konuştuğun kişinin adı Nihal —
ona "Nihal" diye hitap et (kendi adın olan nayh_llia'yı ona hitap ederken
KULLANMA, o senin kendi adın, onun adı değil).

Görevlerin:
- Nihal ile doğal ve samimi bir şekilde sohbet etmek
- Üniversite derslerinde (her konuda) yardımcı olmak, konuları açıklamak
- Beslenme konusunda pratik, sağlıklı öneriler vermek (kesin tıbbi teşhis koymamak)
- Spor/egzersiz konusunda program ve motivasyon önerileri sunmak
- Yabancı dil pratiği yapmasına yardımcı olmak (konuşma alıştırması, düzeltme, çeviri)

Cevapların net, samimi ve gereksiz uzun olmayan bir dilde olsun. SADECE TÜRKÇE
konuş — cevaplarına asla İngilizce (veya başka dilden) kelime karıştırma,
kullanıcı başka bir dilde pratik yapmak istediğini AÇIKÇA belirtmedikçe.
Kullanıcı bir dil pratiği istediğinde o dile geçebilirsin, ama normal
sohbette %100 Türkçe kal. Sesli okunacağı için cevapların konuşma diline
uygun, kısa cümlelerle olsun; uzun listeler/markdown kullanma.
"""

MODEL_ADI = "llama-3.3-70b-versatile"

# ------------------------------------------------------------------
# 6) OTURUM HAFIZASI (session_state)
# ------------------------------------------------------------------
if "mesajlar" not in st.session_state:
    # sistem talimatını da mesaj listesinin başına koyuyoruz (OpenAI formatı böyle çalışır)
    st.session_state.mesajlar = [{"role": "system", "content": SISTEM_TALIMATI}]

# ------------------------------------------------------------------
# 7) GEÇMİŞ MESAJLARI EKRANDA GÖSTER (sistem mesajını gösterme)
# ------------------------------------------------------------------
for mesaj in st.session_state.mesajlar:
    if mesaj["role"] == "system":
        continue
    with st.chat_message(mesaj["role"]):
        st.markdown(mesaj["content"])

# ------------------------------------------------------------------
# 8) SESLİ GİRİŞ (mikrofon -> Groq Whisper ile metne çevirme)
# ------------------------------------------------------------------
st.markdown("**🎤 Sesli konuş:**")
ses_kaydi = mic_recorder(
    start_prompt="🎙️ Kaydı başlat",
    stop_prompt="⏹️ Kaydı durdur",
    key="mikrofon",
)

sesten_gelen_metin = None
if ses_kaydi is not None:
    try:
        # Kaydedilen sesi geçici bir dosyaya yazıp Groq'un Whisper modeline gönderiyoruz
        with open("gecici_kayit.wav", "wb") as f:
            f.write(ses_kaydi["bytes"])
        with open("gecici_kayit.wav", "rb") as f:
            transkript = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                language="tr",
            )
        sesten_gelen_metin = transkript.text
    except Exception as e:
        st.warning(f"Ses işlenemedi: {e}")

# ------------------------------------------------------------------
# 9) YENİ MESAJ AL VE CEVAPLA (yazarak ya da sesle)
# ------------------------------------------------------------------
yazili_girdi = st.chat_input("nayh_llia'ya bir şey sor...")
kullanici_girdisi = yazili_girdi or sesten_gelen_metin

if kullanici_girdisi:
    st.session_state.mesajlar.append({"role": "user", "content": kullanici_girdisi})
    with st.chat_message("user"):
        st.markdown(kullanici_girdisi)

    with st.chat_message("assistant"):
        try:
            cevap = client.chat.completions.create(
                model=MODEL_ADI,
                messages=st.session_state.mesajlar,
            )
            cevap_metni = cevap.choices[0].message.content
            st.markdown(cevap_metni)

            # Cevabı sesli okut (gTTS ile)
            try:
                ses = gTTS(text=cevap_metni, lang="tr")
                ses.save("gecici_cevap.mp3")
                st.audio("gecici_cevap.mp3", format="audio/mp3", autoplay=True)
            except Exception as e:
                st.caption(f"(Ses oluşturulamadı: {e})")

        except Exception as e:
            cevap_metni = f"⚠️ Bir hata oluştu: {e}"
            st.error(cevap_metni)

    st.session_state.mesajlar.append({"role": "assistant", "content": cevap_metni})
