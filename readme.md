# 🎙️ AI Comment Analyzer

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An intelligent, BERT-based application that analyzes text or voice comments and predicts both **emotions** and **toxicity**. Ideal for content moderation, sentiment analysis, and social media monitoring.

## 🚀 Features

- **Multi-Modal Input:** Analyze text comments directly or use your microphone for live voice input.
- **Emotion Classification:** Predicts primary emotions using a fine-tuned BERT model:
  - `Anger` | `Fear` | `Joy` | `Surprise` | `Sadness` | `Love`
- **Toxicity Detection (Multi-Label):** Classifies toxic content into multiple categories:
  - `Toxic` | `Severe Toxic` | `Obscene` | `Threat` | `Insult` | `Identity Hate`
- **GPU Acceleration:** Automatically leverages CUDA if a compatible GPU is available.
- **Clean Text Processing:** Built-in text preprocessing to handle URLs, mentions, hashtags, and special characters.
- **Logging:** Saves all analysis results automatically to `results.txt`.

## 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **HuggingFace Transformers**
- **SpeechRecognition & PyAudio** (for voice input)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Dinesh-luffy/AI-COMMENT-ANALYZER-.git
   cd AI-COMMENT-ANALYZER-
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   > **Note for Windows Users (Microphone Input):**
   > If `pyaudio` fails to install via pip, you may need to install it manually using a precompiled wheel. Download the appropriate wheel for your Python version from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) and install it using `pip install <filename>.whl`.

## 🤖 Model Setup

Due to GitHub file-size constraints, the BERT models are not included directly in this repository. You must supply your own fine-tuned models or modify the code to pull from HuggingFace.

Create a `models` directory in the root of the project and place your models inside:

```text
AI-COMMENT-ANALYZER-/
│
├── comment_analyser.py
├── requirements.txt
├── readme.md
└── models/
    ├── bert_emotion_model/
    └── bert_multi_toxic_model/
```

Inside each model folder, you should have the standard HuggingFace files (`config.json`, `pytorch_model.bin` or `model.safetensors`, `tokenizer_config.json`, `vocab.txt`, etc.).

## ▶️ Usage

Run the main application script:

```bash
python comment_analyser.py
```

You will be greeted with an interactive prompt:
```text
Type '1' to input text, '2' for voice, or 'exit' to quit:
```

### Example Output
```text
 Original: I hate you so much but I love this movie.
 Emotion: Surprise
 Toxicity: Toxic | Insult
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check [issues page](https://github.com/Dinesh-luffy/AI-COMMENT-ANALYZER-/issues).

## 📄 License

This project is licensed under the MIT License.