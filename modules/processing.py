from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib

class TopicExtractor:
    def __init__(self, n_topics=5):
        self.n_topics = n_topics
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.95)
        self.nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=1000)

    def extract(self, input_path, output_csv, output_plot):
        print(f"Extracting topics from {input_path}...")
        df = pd.read_csv(input_path)

        X_tfidf = self.tfidf_vectorizer.fit_transform(df["processed_text"])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        nmf_topics = self.nmf_model.fit_transform(X_tfidf)

        topics = self._display_topics(feature_names)

        df['dominant_topic'] = np.argmax(nmf_topics, axis=1) + 1
        df['topic_confidence'] = np.max(nmf_topics, axis=1)

        for i in range(1, self.n_topics + 1):
            topic_words = topics[f"Topic {i}"]
            mask = df['dominant_topic'] == i
            df.loc[mask, 'topic_keywords'] = ', '.join(topic_words[:5])

        print(f"Saving extracted topics to {output_csv}...")
        df.to_csv(output_csv, index=False)

        self._plot_distribution(df, output_plot)

        return df

    def _display_topics(self, feature_names, n_top_words=10):
        topics = {}
        for topic_idx, topic in enumerate(self.nmf_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics[f"Topic {topic_idx+1}"] = top_words
            print(f"Topic {topic_idx+1}: {' '.join(top_words)}")
        return topics

    def _plot_distribution(self, df, output_plot):
        topic_distribution = df['dominant_topic'].value_counts().sort_index()
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(x=topic_distribution.index, y=topic_distribution.values)
        plt.title('Topic Distribution')
        plt.xlabel('Topic Number')
        plt.ylabel('Document Count')

        for i, v in enumerate(topic_distribution.values):
            ax.text(i, v + 5, str(v), ha='center', fontsize=14)

        plt.tight_layout()
        plt.savefig(output_plot)
        print(f"Topic distribution plot saved to {output_plot}")

class SentimentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, min_df=5, max_df=0.95)),
            ('svm', SVC(kernel='linear', C=1.0, random_state=42, probability=True))
        ])

    def train_and_evaluate(self, input_path, output_prefix, test_size=0.3):
        print(f"Training SVM Classifier using {input_path}...")
        df = pd.read_csv(input_path)

        if 'label' not in df.columns:
            raise ValueError(f"Label column not found in {input_path}")

        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['label'],
            test_size=test_size, random_state=42, stratify=df['label']
        )

        print(f"Training data size: {len(X_train)}, Testing data size: {len(X_test)}")

        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        model_filename = f"{output_prefix}_svm_model.joblib"
        joblib.dump(self.pipeline, model_filename)
        print(f"Model saved to {model_filename}")

        test_results = pd.DataFrame({
            'text': X_test.values,
            'actual_label': y_test.values,
            'predicted_label': y_pred
        })

        class_names = self.pipeline.classes_
        for i, class_name in enumerate(class_names):
            test_results[f'prob_{class_name}'] = [proba[i] for proba in y_proba]

        test_results['is_correct'] = test_results['actual_label'] == test_results['predicted_label']

        test_results_filename = f"{output_prefix}_classification_results.csv"
        test_results.to_csv(test_results_filename, index=False)
        print(f"Test results saved to {test_results_filename}")

        self._plot_confusion_matrix(y_test, y_pred, output_prefix)

        return self.pipeline, test_results

    def _plot_confusion_matrix(self, y_true, y_pred, output_prefix):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        cm_filename = f"{output_prefix}_confusion_matrix.png"
        plt.savefig(cm_filename)
        print(f"Confusion matrix saved to {cm_filename}")

    def _plot_prediction_distribution(self, y_pred, output_prefix):
        prediction_counts = pd.Series(y_pred).value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Prediction Label Distribution')
        plt.axis('equal')

        # Ganti direktori simpan ke /static/assets/plots
        plot_path = f"static/assets/plots/{output_prefix.split('/')[-1]}_prediction_distribution.png"
        plt.savefig(plot_path)
        print(f"Prediction distribution pie chart saved to {plot_path}")



if __name__ == "__main__":
    extractor = TopicExtractor(n_topics=5)
    df_extracted = extractor.extract(
        "data/playstore_preprocessed.csv",
        "data/playstore_extraction.csv",
        "../static/plots/playstore_topic.png"
    )

    classifier = SentimentClassifier()
    model, results = classifier.train_and_evaluate(
        "data/playstore_extraction.csv",
        "output/playstore"
    )
