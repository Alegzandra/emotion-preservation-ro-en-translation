import os
import pandas as pd
from datasets import load_dataset
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm
from collections import Counter

# we use for our experiments: gpt-4o, gpt-4o-mini, gpt-5-mini
# for gpt-5-mini we run this 5 five times, because gpt-5 does not allow modifying the temperature
deployment_name = "gpt-4o-mini"

API_KEY = "******************************************"
client = OpenAI(api_key=API_KEY)

# ===== 2. Emotion Labels =====
emotion_labels = ['Sadness', 'Surprise', 'Fear', 'Anger', 'Neutral', 'Trust', 'Joy']

# ===== 3. Load REDv2 Test Set =====
# we load for our experiments: REDv2 (original in romanian), REDv2EN (english translatd with google translate), REDv2_NLLB (english translated with NLLB)
dataset = load_dataset("Alegzandra/REDv2_NLLB", split="test")

# ===== 4. Prompt Template =====
prompt_template = f"""
You are an expert emotion detection system. Analyze the following sentence and assign one or more emotions from the list: {', '.join(emotion_labels)}.

Respond ONLY with a list of emotions separated by commas, and do not include any explanation or extra words.
"""

# ===== 5. Majority Voting Function =====
def majority_vote_predictions(predictions_list):
    """
    Takes a list of comma-separated emotion predictions from multiple runs
    and returns the most common set.
    """
    # Convert each prediction into a sorted tuple of emotions
    normalized_preds = [tuple(sorted([p.strip() for p in pred.split(",") if p.strip()]))
                        for pred in predictions_list]
    # Count occurrences
    most_common = Counter(normalized_preds).most_common(1)[0][0]
    return ", ".join(most_common)

# ===== 6. Number of runs for each sentence =====
N_RUNS = 1  # Increase for more stability

# ===== 7. Run Predictions with Majority Voting =====
results = []

for item in tqdm(dataset, desc=f"Processing with GPT-4o-mini ({N_RUNS} runs each)"):
    sentence = item["text"]
    user_prompt = f"{prompt_template}\n\nSentence: {sentence}"
    preds = []

    for _ in range(N_RUNS):
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for emotion detection."},
                    {"role": "user", "content": user_prompt}
                ]
            )
            prediction = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Error on run for: {sentence[:50]}... - {e}")
            prediction = ""
        preds.append(prediction)

    # Apply majority vote to get final stable prediction
    final_prediction = majority_vote_predictions(preds)


    results.append({
        "Sentence": sentence,
        "Predicted Emotions": final_prediction
    })

# ===== 8. Save Predictions to Excel =====
df = pd.DataFrame(results)
df.to_excel("REDv2_EN_NLLB_GPT4o_mini_MajorityVote_Predictions run 0.xlsx", index=False)
#print("Saved GPT-5-mini predictions with majority voting to REDv2_GPT5nano_MajorityVote_Predictions.xlsx")
