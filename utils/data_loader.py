import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
from config import Config

scraped_data = None
preprocessed_data = None
classification_results = None
svm_model = None
tfidf_vectorizer = None
X_tfidf = None
nmf_model = None
topic_distribution = None
feature_names = None

def load_all_data():
    global scraped_data, preprocessed_data, classification_results
    global svm_model, tfidf_vectorizer, X_tfidf, nmf_model, topic_distribution, feature_names

    scraped_data = pd.read_csv(Config.RAW_DATA)
    preprocessed_data = pd.read_csv(Config.PREPROCESSED_DATA)
    classification_results = pd.read_csv(Config.CLASSIFICATION_RESULTS)
    svm_model = joblib.load(Config.MODEL_PATH)

    tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.95)
    X_tfidf = tfidf_vectorizer.fit_transform(preprocessed_data["processed_text"])
    feature_names = tfidf_vectorizer.get_feature_names_out()

    nmf_model = NMF(n_components=5, random_state=42, max_iter=1000)
    nmf_topics = nmf_model.fit_transform(X_tfidf)
    preprocessed_data['dominant_topic'] = np.argmax(nmf_topics, axis=1) + 1
    topic_distribution = preprocessed_data['dominant_topic'].value_counts().sort_index()
