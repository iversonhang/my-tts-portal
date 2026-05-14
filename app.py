import streamlit as st
import io
import os
from google import genai
from google.genai import types
from pydub import AudioSegment

# 1. Page Configuration & Styling
st.set_page_config(page_title="Gemini TTS Portal", page_icon="🎙️", layout="centered")

st.title("🎙️ Gemini Multilingual TTS Portal")
st.caption("Convert text to high-fidelity MP3 speech natively using Gemini AI.")

# 2. Initialize Gemini Client safely
# It automatically picks up the GEMINI_API_KEY environment variable
try:
    client = genai.Client()
except Exception as e:
    st.error("Failed to initialize Gemini Client. Please ensure your GEMINI_API_KEY environment variable is set.")
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
if st.button("Synthesize to MP3", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text before synthesizing.")
    else:
        with st.spinner("Generating audio performance..."):
            full_prompt = f"{config['prompt_prefix']}\n\n{text_input}"
            
            try:
                # Requesting audio generation from the Gemini TTS capable model
                response = client.models.generate_content(
                    model="gemini-3.1-flash",
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
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        audio_bytes = part.inline_data.data
                        break
                
                if not audio_bytes:
                    st.error("No audio data was returned by the Gemini API.")
                else:
                    # Convert raw/wav response data to high-quality MP3 using Pydub
                    wav_io = io.BytesIO(audio_bytes)
                    audio_segment = AudioSegment.from_file(wav_io)
                    
                    mp3_io = io.BytesIO()
                    audio_segment.export(mp3_io, format="mp3", bitrate="192k")
                    mp3_data = mp3_io.getvalue()

                    # 6. Render Output Audio Player and Download Options
                    st.success("Audio Generated Successfully!")
                    
                    # Native Streamlit Audio Player
                    st.audio(mp3_data, format="audio/mp3")
                    
                    # Native Streamlit Download Button
                    st.download_button(
                        label="📥 Download MP3 File",
                        data=mp3_data,
                        file_name=f"gemini_speech_{config['code']}.mp3",
                        mime="audio/mpeg",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")