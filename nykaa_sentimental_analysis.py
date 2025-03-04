from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
# Function to plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# Function to plot ROC Curve
def plot_roc_curve(y_true, y_pred_proba, model_name):
    plt.figure(figsize=(7, 6))
    for i, class_label in enumerate(label_encoder.classes_):
        fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
        plt.plot(fpr, tpr, label=f"{class_label} (AUC: {auc(fpr, tpr):.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.show()

# Function to plot Precision-Recall Curve
def plot_precision_recall(y_true, y_pred_proba, model_name):
    plt.figure(figsize=(7, 6))
    for i, class_label in enumerate(label_encoder.classes_):
        precision, recall, _ = roc_curve(y_true == i, y_pred_proba[:, i])
        plt.plot(recall, precision, label=f"{class_label}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend()
    plt.show()

# Generate visualizations for each model
for model_name, model in optimized_models.items():
    print(f"Evaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test_tfidf_sampled)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test_encoded_sampled, y_pred, model_name)
    
    # Plot ROC and Precision-Recall curves if probability predictions are available
    if model_name in y_pred_proba_optimized_fast and y_pred_proba_optimized_fast[model_name] is not None:
        plot_roc_curve(y_test_encoded_sampled, y_pred_proba_optimized_fast[model_name], model_name)
        plot_precision_recall(y_test_encoded_sampled, y_pred_proba_optimized_fast[model_name], model_name)
