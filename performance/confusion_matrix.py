"""
Qualitative performance.
"""
import pandas as pd


def main(results_df):
    """
    Main function for confusion matrix.
    """
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for row in results_df.itertuples():
        label = int(row[1])
        prediction = int(row[2])
        if prediction == 1:
            if label == 1:
                true_pos += 1
            elif label == -1:
                false_pos += 1
        elif prediction == -1:
            if label == 1:
                false_neg += 1
            elif label == -1:
                true_neg += 1
    matrix = pd.DataFrame(
        [
            [true_pos, false_pos, int(true_pos+false_pos)],
            [false_neg, true_neg, int(false_neg+true_neg)],
            [int(true_pos+false_neg), int(false_pos+true_neg), 0]
        ],
        columns=["real pos", "real neg", "total predicted"],
        index=["predict pos", "predict neg", "total real"]
    )
    # Total
    pos = true_pos + false_neg
    neg = true_neg + false_pos
    # Recall
    recall = true_pos/(pos+1)
    # Specificity
    spec = true_pos/(neg+1)
    # Precision
    precision = true_pos/(true_pos+false_pos+1)
    # Precision negative
    npv = true_neg/(true_neg+false_neg+1)
    # Accuracy
    acc = (true_pos+true_neg)/(pos+neg+1)
    # Balanced accuracy
    bacc = (true_pos/(pos+1)+true_neg/(neg+1))/2
    # F1 score
    f1_score = precision*recall/(precision+recall+1)
    return {
        "confusion_matrix": matrix,
        "recall": round(recall, 2),
        "specificity": round(spec, 2),
        "precision": round(precision, 2),
        "negative predictive": round(npv, 2),
        "accuracy": round(acc, 2),
        "balanced accuracy": round(bacc, 2),
        "f_score": round(f1_score, 2)
    }
