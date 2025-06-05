import os

class Config:
    PLOT_DIR = 'static/plots'
    RAW_DATA = 'data/playstore.csv'
    PREPROCESSED_DATA = 'data/playstore_preprocessed.csv'
    CLASSIFICATION_RESULTS = 'data/playstore_classification_results.csv'
    MODEL_PATH = 'model/playstore_svm_model.joblib'

    @staticmethod
    def init_dirs():
        os.makedirs(Config.PLOT_DIR, exist_ok=True)
