import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ===== 1. Load REDv2 test set =====
dataset = load_dataset("Alegzandra/REDv2", split="test")

# ===== 2. Emotion label order =====
emotion_labels = ['Sadness', 'Surprise', 'Fear', 'Anger', 'Neutral', 'Trust', 'Joy']

# ===== 3. Load prediction files =====
# Make sure these Excel files have columns: Sentence, Predicted Emotions
df_gpt4o = pd.read_excel("REDv2_GPT4o_Predictions.xlsx")

df_gpt4o_mini = pd.read_excel("REDv2_GPT4o_mini_Predictions.xlsx")

df_gpt5 = pd.read_excel("REDv2_EN_GPT5mini_MajorityVote_Predictions run 1.xlsx")  # GPT-5 predictions


# ===== 4. Convert emotions string to binary vector =====
def emotions_to_vector(emotions_str):
    if pd.isna(emotions_str) or not emotions_str.strip():
        return [0] * len(emotion_labels)
    pred_emotions = [e.strip() for e in emotions_str.split(",") if e.strip()]
    return [1 if label in pred_emotions else 0 for label in emotion_labels]

# ===== 5. Prepare ground truth and prediction arrays =====
y_true, y_gpt4o, y_gpt4o_mini, y_gpt5 = [], [], [], []
merged_rows = []

for item in dataset:
    sentence = item['text']
    # Ground truth binary vector
    gt_vector = item['agreed_labels']
    y_true.append(gt_vector)
    gt_labels = ', '.join([lbl for lbl, val in zip(emotion_labels, gt_vector) if val == 1])

    # Helper to fetch predictions from a dataframe
    def get_prediction(df):
        match = df.loc[df['Sentence'] == sentence, 'Predicted Emotions']
        return str(match.values[0]).strip() if not match.empty else ''

    p4o = get_prediction(df_gpt4o)
    p4o_mini = get_prediction(df_gpt4o_mini)
    p5 = get_prediction(df_gpt5)

    y_gpt4o.append(emotions_to_vector(p4o))
    y_gpt4o_mini.append(emotions_to_vector(p4o_mini))
    y_gpt5.append(emotions_to_vector(p5))

    merged_rows.append({
        "Sentence": sentence,
        "Ground Truth": gt_labels,
        "GPT-4o": p4o,
        "GPT-4o-mini": p4o_mini,
        "GPT-5": p5
    })

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_gpt4o = np.array(y_gpt4o)
y_gpt4o_mini = np.array(y_gpt4o_mini)
y_gpt5 = np.array(y_gpt5)

# ===== 6. Metric computation =====
def compute_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    try:
        auc_macro = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        auc_macro = None

    print(f"\n📊 Metrics for {model_name}:")
    print(f"  ✅ Accuracy (subset): {acc:.4f}")
    print(f"  ✅ F1-score (macro): {f1_macro:.4f}")
    print(f"  ✅ F1-score (micro): {f1_micro:.4f}")
    if auc_macro is not None:
        print(f"  ✅ ROC AUC (macro): {auc_macro:.4f}")
    else:
        print("  ⚠️ ROC AUC could not be computed.")

# ===== 7. Show results for each model =====
compute_metrics(y_true, y_gpt4o, "GPT-4o")
compute_metrics(y_true, y_gpt4o_mini, "GPT-4o-mini")
compute_metrics(y_true, y_gpt5, "GPT-5")

# ===== 8. Save merged comparison Excel =====
merged_df = pd.DataFrame(merged_rows)
merged_df.to_excel("REDv2_Comparisons_with_GPT_new.xlsx", index=False)
print("\n💾 Merged results saved to REDv2_Comparisons_with_GPT.xlsx")
