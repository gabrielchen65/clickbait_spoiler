import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score
import sys


# Load the CSV file into a pandas DataFrame
file_path = sys.argv[1]
data = pd.read_csv(file_path)

# Assuming the 2nd column is 'y_pred' and the 3rd column is 'y_true'
y_pred = data.iloc[:, 1]
y_true = data.iloc[:, 2]

# Calculate precision, recall, and F1 score
micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("Micro Precision:", micro_precision)
print("Micro Recall:", micro_recall)
print("Micro F1 Score:", micro_f1)
print("Balanced accuracy:",balanced_accuracy_score(y_true, y_pred))
print()


precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=y_true.unique())

# Get unique class labels
class_labels = y_true.unique()

# Create a dictionary to store metrics for each class
class_metrics = {}
for label, p, r, f in zip(class_labels, precision, recall, f1):
    class_metrics[label] = {
        'Precision': p,
        'Recall': r,
        'F1 Score': f
    }

# Print metrics for each class
for label, metrics in class_metrics.items():
    print(f"Metrics for Class {label}:")
    print("Precision:", metrics['Precision'])
    print("Recall:", metrics['Recall'])
    print("F1 Score:", metrics['F1 Score'])
    print()




