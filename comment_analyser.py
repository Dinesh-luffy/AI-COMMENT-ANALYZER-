from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
import os
import speech_recognition as sr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model paths
sentiment_model_path = "bert_emotion_model"
toxic_model_path = "bert_multi_toxic_model"

# Verify model directories
if not os.path.isdir(sentiment_model_path):
    raise FileNotFoundError(f" Sentiment model folder not found: {sentiment_model_path}")
if not os.path.isdir(toxic_model_path):
    raise FileNotFoundError(f" Toxic model folder not found: {toxic_model_path}")

# Load models and tokenizers
sentiment_tokenizer = BertTokenizer.from_pretrained(sentiment_model_path, local_files_only=True)
sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_path, local_files_only=True)
sentiment_model.to(device).eval()

toxic_tokenizer = BertTokenizer.from_pretrained(toxic_model_path, local_files_only=True)
toxic_model = BertForSequenceClassification.from_pretrained(toxic_model_path, local_files_only=True)
toxic_model.to(device).eval()

# Labels
sentiment_labels = ['Anger', 'Fear', 'Joy', 'Surprise', 'Sadness', 'Love']
toxic_labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']


def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]+", "", text)
    return text.strip().lower()


def analyze_comment(comment):
    cleaned = clean_text(comment)

    # Emotion prediction
    s_inputs = sentiment_tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    s_inputs = {k: v.to(device) for k, v in s_inputs.items()}
    s_output = sentiment_model(**s_inputs)
    sentiment = torch.argmax(s_output.logits, dim=-1).item()
    emotion_label = sentiment_labels[sentiment]

    # Toxicity prediction
    t_inputs = toxic_tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    t_inputs = {k: v.to(device) for k, v in t_inputs.items()}
    t_output = toxic_model(**t_inputs)
    probs = torch.sigmoid(t_output.logits).detach().cpu().numpy()[0]
    toxic_result = [toxic_labels[i] for i, prob in enumerate(probs) if prob >= 0.5]

    # Display result
    print("\n Original:", comment)
    print(" Emotion:", emotion_label)
    if toxic_result:
        print(" Toxicity:", " | ".join(toxic_result))
    else:
        print(" Toxicity: Clean (No toxic content)")


def get_voice_input():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("\n Speak now (Listening up to 30 seconds)...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)

    try:
        print(" Recognizing...")
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print(" Could not understand audio.")
    except sr.RequestError:
        print(" Could not request results from Google Speech Recognition.")
    return None


if __name__ == "__main__":
    print(" Comment Emotion + Toxicity Analyzer Ready!")
    while True:
        choice = input("\nType '1' to input text, '2' for voice, or 'exit' to quit: ").strip().lower()
        if choice == "exit":
            print("👋 Exiting Comment Analyzer.")
            break
        elif choice == "1":
            text = input("📝 Enter your comment: ")
            analyze_comment(text)
        elif choice == "2":
            text = get_voice_input()
            if text:
                print(f"📝 You said: {text}")
                analyze_comment(text)
        else:
            print("❗ Invalid choice. Please type '1', '2', or 'exit'.")
