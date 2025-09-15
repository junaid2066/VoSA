import os
import datetime
import streamlit as st
import whisper
import numpy as np
import soundfile as sf
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
import librosa

# Load the Whisper model
model = whisper.load_model("large")  # Load a larger model for better accuracy

# Initialize the sentiment analysis pipeline with the new model
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="tabularisai/multilingual-sentiment-analysis"
)

# Function to analyze sentiment
def analyze_sentiment(text):
    results = sentiment_analysis(text)
    sentiment_results = {result['label']: result['score'] for result in results}
    return sentiment_results

# Function to analyze emotions based on keywords
def analyze_emotions(text):
    emotion_keywords = {
        "joy": ["happy", "joy", "excited", "great", "fantastic", "amazing"],
        "sadness": ["sad", "unhappy", "sorrow", "down", "depressed"],
        "anger": ["angry", "mad", "frustrated", "hate", "annoyed"],
        "fear": ["fear", "scared", "afraid", "terrified", "worried"],
        "surprise": ["surprised", "shocked", "unexpected", "astonished"],
        "disgust": ["disgusted", "gross", "sick", "yuck"]
    }
    
    detected_emotions = {}
    words = text.lower().split()
    
    # Check for keywords in the text
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in words:
                detected_emotions[emotion] = detected_emotions.get(emotion, 0) + 1

    return detected_emotions

# Function to get sentiment emoji
def get_sentiment_emoji(sentiment):
    emoji_mapping = {
        "disappointment": "😞",
        "sadness": "😢",
        "annoyance": "😠",
        "neutral": "😐",
        "disapproval": "👎",
        "realization": "😮",
        "nervousness": "😬",
        "approval": "👍",
        "joy": "😄",
        "anger": "😡",
        "embarrassment": "😳",
        "caring": "🤗",
        "remorse": "😔",
        "disgust": "🤢",
        "grief": "😥",
        "confusion": "😕",
        "relief": "😌",
        "desire": "😍",
        "admiration": "😌",
        "optimism": "😊",
        "fear": "😨",
        "love": "❤️",
        "excitement": "🎉",
        "curiosity": "🤔",
        "amusement": "😄",
        "surprise": "😲",
        "gratitude": "🙏",
        "pride": "🦁"
    }
    return emoji_mapping.get(sentiment, "")

# Function to display sentiment results
def display_sentiment_results(sentiment_results, emotion_results, option):
    sentiment_text = ""
    for sentiment, score in sentiment_results.items():
        emoji = get_sentiment_emoji(sentiment)
        if option == "Sentiment Only":
            sentiment_text += f"{sentiment} {emoji}\n"
        elif option == "Sentiment + Score":
            sentiment_text += f"{sentiment} {emoji}: {score:.2f}\n"
    
    # Display detected emotions
    if emotion_results:
        sentiment_text += "\n**Detected Emotions:**\n"
        for emotion, count in emotion_results.items():
            sentiment_text += f"{emotion.capitalize()}: {count} occurrence(s)\n"

    return sentiment_text

# Function to save audio file
def save_audio_file(audio_bytes, file_extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"audio_{timestamp}.{file_extension}"
    
    with open(file_name, "wb") as f:
        f.write(audio_bytes)

    return file_name

# Function to transcribe audio
def transcribe_audio(file_path, model):
    audio_data, sample_rate = librosa.load(file_path, sr=16000, mono=True)  # Resample to 16 kHz
    transcript = model.transcribe(audio_data, fp16=False, verbose=True)
    return transcript["text"]

# Function to display transcript
def display_transcript(transcript_text):
    st.subheader("Transcription Result")
    st.write(transcript_text)

# Main function to run the app
def main():
    st.markdown("<h2 style='font-size: 40px;'>🎤 VoSA: Voice Sentiment Analyzer 💬</h2>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])

    # Record Audio tab
    with tab1:
        audio_bytes = audio_recorder()
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            file_name = save_audio_file(audio_bytes, "wav")
            st.success(f"Recorded audio saved as: {file_name}")

    # Upload Audio tab
    with tab2:
        audio_file = st.file_uploader("Upload Audio", type=["mp3", "mp4", "wav", "m4a"])
        if audio_file:
            file_extension = audio_file.type.split('/')[1]
            file_name = save_audio_file(audio_file.read(), file_extension)
            st.success(f"Uploaded audio saved as: {file_name}")

    # Sentiment option
    sentiment_option = st.radio("Select sentiment analysis option", ["Sentiment Only", "Sentiment + Score"])

    # Transcribe button action
    if st.button("Process your Audio"):
        try:
            audio_files = [f for f in os.listdir(".") if f.startswith("audio")]
            if not audio_files:
                st.error("No audio files found. Please record or upload audio first.")
                return

            audio_file_path = max(audio_files, key=os.path.getctime)
            transcript_text = transcribe_audio(audio_file_path, model)
            display_transcript(transcript_text)

            # Analyze sentiment and emotions
            sentiment_results = analyze_sentiment(transcript_text)
            emotion_results = analyze_emotions(transcript_text)
            sentiment_output = display_sentiment_results(sentiment_results, emotion_results, sentiment_option)

            # Display sentiment results
            st.write("**Sentiment and Emotion Analysis Results:**")
            st.text(sentiment_output)
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
    # Footer
    st.markdown("<footer style='text-align: center; font-size: 14px;'>Developed by Natural Language Processing Lab</footer>", unsafe_allow_html=True)
    st.markdown("<footer style='text-align: center; font-size: 12px;'>Under the umbrella of Artificial Intelligence Technology Centre (AITeC)</footer>", unsafe_allow_html=True)
    st.markdown("<footer style='text-align: center; font-size: 10px;'>National Centre for Physics (NCP)</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()