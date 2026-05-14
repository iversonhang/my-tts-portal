import streamlit as st
import io
import wave
from google import genai
from google.genai import types

# 1. Page Configuration & Styling
st.set_page_config(page_title="Gemini TTS Portal", page_icon="🎙️", layout="centered")

st.title("🎙️ Gemini Multilingual TTS Portal")
st.caption("Convert text to high-fidelity, universally playable audio using Gemini AI.")

# 2. Initialize Gemini Client safely
try:
    client = genai.Client()
except Exception as e:
    st.error("Failed to initialize Gemini Client. Please ensure your GEMINI_API_KEY environment variable is set in your Streamlit Secrets.")
    st.stop()

# 3. Setup Language Configurations with Explicit BCP-47 Language Codes
LANGUAGE_CONFIGS = {
    "English (Global/US)": {
        "code": "en",
        "lang_code": "en-US",
        "prompt_prefix": "Please read the following text naturally, clearly, and with an accurate English accent: ",
        "voice": "Aoede"
    },
    "Mandarin (普通话)": {
        "code": "zh-CN",
        "lang_code": "zh-CN",
        "prompt_prefix": "请用标准、地道的普通话清晰自然地朗读以下文字：",
        "voice": "Puck"
    },
    "Cantonese (粵語)": {
        "code": "zh-HK",
        "lang_code": "zh-HK",
        "prompt_prefix": "請用標準、純正的廣東話（粵語）地道發音朗讀以下文字：",
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
                    model="gemini-2.5-flash-preview-tts",  # <-- Swapped from 3.1-flash-tts-preview
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            # FIX: Pass the required BCP-47 language code to force the native engine accent
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
                    st.error("No audio data was returned by the Gemini API. Please verify the input text style.")
                else:
                    # Wrap raw 24kHz, 16-bit, Mono PCM bytes into a standard WAV header container
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, 'wb') as wav_file:
                        wav_file.setnchannels(1)      # Mono
                        wav_file.setsampwidth(2)      # 16-bit (2 bytes per sample)
                        wav_file.setframerate(24000)  # 24kHz
                        wav_file.writeframes(raw_pcm_bytes)
                    
                    playable_wav_data = wav_buffer.getvalue()

                    st.success(f"Audio Generated Successfully in {selected_lang_name}!")
                    
                    # Native Streamlit Audio Player
                    st.audio(playable_wav_data, format="audio/wav")
                    
                    # Native Streamlit Download Button
                    st.download_button(
                        label="📥 Download Audio File (WAV)",
                        data=playable_wav_data,
                        file_name=f"gemini_speech_{config['code']}.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
