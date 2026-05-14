import streamlit as st
import io
import wave
from google import genai
from google.genai import types

# 1. Page Configuration & Styling
st.set_page_config(page_title="Gemini TTS Portal", page_icon="🎙️", layout="centered")

st.title("🎙️ Gemini Multilingual TTS Portal")
st.caption("Convert language-specific text to high-fidelity audio using independent tabs.")

# 2. Initialize Gemini Client safely
try:
    client = genai.Client()
except Exception as e:
    st.error("Failed to initialize Gemini Client. Please ensure your GEMINI_API_KEY environment variable is set in your Streamlit Secrets.")
    st.stop()

# 3. Setup Language Configurations
LANGUAGE_CONFIGS = {
    "English": {
        "code": "en",
        "lang_code": "en-US",
        "prompt_prefix": "Please read the following text naturally, clearly, and with an accurate English accent: ",
        "voice": "Aoede",
        "placeholder": "Type your English text here..."
    },
    "Mandarin (普通话)": {
        "code": "zh-CN",
        "lang_code": "zh-CN",
        "prompt_prefix": "请用标准、地道的普通话清晰自然地朗读以下文字：",
        "voice": "Puck",
        "placeholder": "在这里输入简体中文/普通话文本..."
    },
    "Cantonese (粵語)": {
        "code": "zh-HK",
        "lang_code": "zh-HK",
        "prompt_prefix": "請用標準、純正的廣東話（粵語）地道發音朗讀以下文字：",
        "voice": "Puck",
        "placeholder": "在這裡輸入繁體中文/粵語文本..."
    }
}

# 4. Creating the Core Logic Function
def process_tts(text_input, config, lang_label):
    if not text_input.strip():
        st.warning(f"Please enter some text in the {lang_label} tab before synthesizing.")
        return

    with st.spinner(f"Generating {lang_label} audio performance..."):
        full_prompt = f"{config['prompt_prefix']}\n\n{text_input}"
        
        try:
            # Requesting audio generation from the Gemini TTS model
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",  # <-- Swapped from 3.1-flash-tts-preview
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        language_code=config["lang_code"],
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=config["voice"]
                            )
                        )
                    )
                )
            )
            
            # Extract inline raw PCM audio binary data
            raw_pcm_bytes = None
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        raw_pcm_bytes = part.inline_data.data
                        break
            
            if not raw_pcm_bytes:
                st.error("No audio data was returned by the Gemini API.")
            else:
                # Wrap raw 24kHz, 16-bit, Mono PCM bytes into a standard WAV header container
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)      # Mono
                    wav_file.setsampwidth(2)      # 16-bit
                    wav_file.setframerate(24000)  # 24kHz
                    wav_file.writeframes(raw_pcm_bytes)
                
                playable_wav_data = wav_buffer.getvalue()

                st.success(f"Successfully generated {lang_label} audio!")
                
                # Render inline audio player
                st.audio(playable_wav_data, format="audio/wav")
                
                # Render standalone download button
                st.download_button(
                    label=f"📥 Download {lang_label} Audio (WAV)",
                    data=playable_wav_data,
                    file_name=f"gemini_speech_{config['code']}.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"An error occurred: {e}")

# 5. Rendering Independent Tabs for each script
tab1, tab2, tab3 = st.tabs(["🇺🇸 English", "🇨🇳 Mandarin (普通话)", "🇭🇰 Cantonese (粵語)"])

with tab1:
    cfg = LANGUAGE_CONFIGS["English"]
    en_text = st.text_area("English Input Script", height=180, placeholder=cfg["placeholder"], key="en_area")
    if st.button("Synthesize English", type="primary", use_container_width=True, key="en_btn"):
        process_tts(en_text, cfg, "English")

with tab2:
    cfg = LANGUAGE_CONFIGS["Mandarin (普通话)"]
    cn_text = st.text_area("Mandarin Input Script", height=180, placeholder=cfg["placeholder"], key="cn_area")
    if st.button("Synthesize Mandarin", type="primary", use_container_width=True, key="cn_btn"):
        process_tts(cn_text, cfg, "Mandarin")

with tab3:
    cfg = LANGUAGE_CONFIGS["Cantonese (粵語)"]
    hk_text = st.text_area("Cantonese Input Script", height=180, placeholder=cfg["placeholder"], key="hk_area")
    if st.button("Synthesize Cantonese", type="primary", use_container_width=True, key="hk_btn"):
        process_tts(hk_text, cfg, "Cantonese")
