# 🎤 VoSA: Voice Sentiment Analyzer 💬

## 📖 Description:
VoSA is an intelligent application that decodes emotions and sentiments from human voice in real-time. It leverages OpenAI’s Whisper for speech-to-text transcription and a multilingual sentiment analysis model to detect polarity and emotional states. The system also performs keyword-based emotion detection and presents results with emoji-rich visualizations for better interpretability.

## ✨ Features:
- 🎙️ Real-time audio recording or file upload (MP3, WAV, MP4, M4A)
- 📝 Accurate speech-to-text transcription (via Whisper)
- 📊 Sentiment analysis with multilingual support
- 😊 Emotion detection (joy, sadness, anger, fear, surprise, disgust)
- ⚡ Interactive Streamlit UI with modern design and institutional branding
- 💡 Useful for psychology, customer feedback, HCI, and education research

## ✨ Models Used:
- **🎙️ Speech-to-Text (STT):** [Open AI Whisper 1500M](https://github.com/openai/whisper)
- **Sentiment Analysis:** [Multilingual Sentiment Analysis Model publicly available on Hugging face supoorting uto 36 languages.](https://huggingface.co/tabularisai/multilingual-sentiment-analysis)
- **For Audio File Handling:** [LIBROSA](https://github.com/librosa/librosa)

## ✨ Installation:
### A. Requirements:
- Python 3.9+
- Whisper Large (1500M)
- Streamlit
- Hugging Face Transformers
- Librosa
- SoundFile

### B. Usage

1. Start the Streamlit app: streamlit run app.py
2. Choose one of the input methods:
- Record audio directly in the browser
- Upload an audio file
3. Click 🚀 Analyze Audio to see:
- Transcription of your speech
- Sentiment analysis results with scores
- Emotion detection with emoji mapping

## 📊 Project Status
🚀 Active Development – new features and improvements coming soon.
