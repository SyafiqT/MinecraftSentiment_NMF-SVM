import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from utils import data_loader

def generate_plots():
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
    x=data_loader.topic_distribution.index,
    y=data_loader.topic_distribution.values,
    hue=data_loader.topic_distribution.index,
    palette='viridis',
    legend=False
)

    plt.title('Topic Distribution')
    plt.xlabel('Topic Number')
    plt.ylabel('Count')
    for i, v in enumerate(data_loader.topic_distribution.values):
        ax.text(i, v + 5, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(f"{Config.PLOT_DIR}/playstore_topic.png")
    plt.close()

    if 'actual_label' in data_loader.classification_results.columns:
        plt.figure(figsize=(8, 6))
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(
            data_loader.classification_results['actual_label'],
            data_loader.classification_results['predicted_label']
        )
        unique_labels = sorted(data_loader.preprocessed_data['label'].unique())
        cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f"{Config.PLOT_DIR}/playstore_confusion_matrix.png")
        plt.close()

def get_topics():
    topics = {}
    for topic_idx, topic in enumerate(data_loader.nmf_model.components_):
        top_words = [data_loader.feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics[f"Topic {topic_idx + 1}"] = top_words
    return topics

topic_distribution = lambda: data_loader.topic_distribution
