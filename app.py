import streamlit as st
import base64
from google import genai
from google.genai import types

# 1. Page Configuration & Styling
st.set_page_config(page_title="Gemini TTS Portal", page_icon="🎙️", layout="centered")

st.title("🎙️ Gemini Multilingual TTS Portal")
st.caption("Convert text to high-fidelity speech natively using Gemini AI (Supports Python 3.13+).")

# 2. Initialize Gemini Client safely
try:
    client = genai.Client()
except Exception as e:
    st.error("Failed to initialize Gemini Client. Please ensure your GEMINI_API_KEY environment variable is set in your Streamlit Secrets.")
    st.stop()

# 3. Setup Language Configurations
LANGUAGE_CONFIGS = {
    "English (Global/US)": {
        "code": "en",
        "prompt_prefix": "[Language: English] Please read the following text naturally and clearly: ",
        "voice": "Aoede"
    },
    "Mandarin (普通话)": {
        "code": "zh-CN",
        "prompt_prefix": "[Language: Mandarin Chinese] 请用标准普通话清晰、自然地朗读以下文字：",
        "voice": "Puck"
    },
    "Cantonese (粵語)": {
        "code": "zh-HK",
        "prompt_prefix": "[Language: Cantonese Chinese] 請用標準粵語及地道發音朗讀以下文字：",
        "voice": "Puck"
    }
}

# 4. Building the UI Components
selected_lang_name = st.selectbox("Target Language", list(LANGUAGE_CONFIGS.keys()))
config = LANGUAGE_CONFIGS[selected_lang_name]

text_input = st.text_area(
    "Input Script", 
    height=200, 
    placeholder="Type or paste your text here...\nTip: You can insert expression tags like [excitedly] or [whispers] to alter emotions dynamically!"
)

# 5. Core Processing Pipeline
if st.button("Synthesize Audio", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text before synthesizing.")
    else:
        with st.spinner("Generating audio performance..."):
            full_prompt = f"{config['prompt_prefix']}\n\n{text_input}"
            
            try:
                # Requesting audio generation from the specific Gemini TTS model
                response = client.models.generate_content(
                    model="gemini-3.1-flash-tts-preview",
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=config["voice"]
                                )
                            )
                        )
                    )
                )
                
                # Extract inline audio binary data
                audio_bytes = None
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            audio_bytes = part.inline_data.data
                            break
                
                if not audio_bytes:
                    st.error("No audio data was returned by the Gemini API. Please verify the input text style.")
                else:
                    st.success("Audio Generated Successfully!")
                    
                    # Native Streamlit Audio Player (Gemini natively outputs high-quality compressed audio)
                    st.audio(audio_bytes, format="audio/mp3")
                    
                    # Native Streamlit Download Button
                    st.download_button(
                        label="📥 Download Audio File",
                        data=audio_bytes,
                        file_name=f"gemini_speech_{config['code']}.mp3",
                        mime="audio/mpeg",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
