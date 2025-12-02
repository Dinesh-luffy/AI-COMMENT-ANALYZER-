from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
import os
import speech_recognition as sr
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sentiment_model_path = os.path.join(BASE_DIR, "models", "bert_emotion_model")
toxic_model_path = os.path.join(BASE_DIR, "models", "bert_multi_toxic_model")

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
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
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

    # Save to log
    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(f"{comment}\nEmotion: {emotion_label}\nToxicity: {toxic_result or 'Clean'}\n\n")


def get_voice_input():
    recognizer = sr.Recognizer()
    
    try:
        mic = sr.Microphone()
    except Exception as e:
        print(f"❗ Microphone not accessible: {e}")
        return None
    
    for attempt in range(2):
        try:
            with mic as source:
                print(f"\n🎤 Listening... (Attempt {attempt + 1}/2)")
                print("   Adjusting for background noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Adjust sensitivity
                recognizer.energy_threshold = 300
                recognizer.dynamic_energy_threshold = True
                
                print("   ✅ Ready! Speak now...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
                
            print("   🔄 Processing speech...")
            text = recognizer.recognize_google(audio)
            return text
            
        except sr.WaitTimeoutError:
            print(f"   ⏱️ No speech detected. Please try again.")
        except sr.UnknownValueError:
            print("   ❓ Could not understand the audio.")
        except sr.RequestError as e:
            print(f"   ❗ Network error: {e}")
        except Exception as e:
            print(f"   ❗ Error: {e}")
    
    print("❌ Voice input failed. Returning to menu...\n")
    return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_comment(" ".join(sys.argv[1:]))
    else:
        while True:
            choice = input("\nType '1' to input text, '2' for voice, or 'exit' to quit: ").strip().lower()
            if choice == "exit":
                print("👋 Exiting Comment Analyzer.")
                break
            elif choice == "1":
                    text = input("📝 Enter your comment: ").strip()
                    if not text or len(text) < 3:
                        print("❗ Comment too short. Try again.")
                        continue
                    analyze_comment(text)

            elif choice == "2":
                text = get_voice_input()
                if text:
                    print(f"📝 You said: {text}")
                    analyze_comment(text)
            else:
                print("❗ Invalid choice. Please type '1', '2', or 'exit'.")
