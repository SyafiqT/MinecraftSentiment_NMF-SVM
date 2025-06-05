import re
import pandas as pd
import nltk
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class Preprocessor:
    def __init__(self, stop_words=None):
        self.stop_words = stop_words or set(stopwords.words('english'))
        
    def cleaning(self, text):
        if isinstance(text, str):
            text = emoji.replace_emoji(text, replace='')  
            text = re.sub(r'[^a-zA-Z\s]', '', text)  
            text = text.strip()
            return text if text else "unknown"
        return "unknown"

    def tokenization(self, text):
        if isinstance(text, str) and text.strip():
            return text.split()
        return ["unknown"]

    def case_folding(self, tokens):
        return [word.lower() for word in tokens if word]

    def remove_stopwords(self, tokens):
        filtered_tokens = [word for word in tokens if word and word not in self.stop_words]
        return filtered_tokens if filtered_tokens else ["unknown"]

    def lemmatization(self, tokens):
        return [TextBlob(word).words[0].lemmatize() if word else "unknown" for word in tokens]

    def sentiment_labeling(self, text):
        if isinstance(text, str) and text.strip():
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0:
                return "positive"
            elif polarity < 0:
                return "negative"
        return "neutral"

    def preprocess(self, input_path, output_path):
        print(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        df.dropna(subset=["Comment"], inplace=True)
        df["Comment"] = df["Comment"].fillna("unknown")

        print("Preprocessing steps...")
        df["cleaned_text"] = df["Comment"].apply(self.cleaning)
        df["tokens"] = df["cleaned_text"].apply(self.tokenization)
        df["tokens"] = df["tokens"].apply(self.case_folding)
        df["tokens"] = df["tokens"].apply(self.remove_stopwords)
        df["tokens"] = df["tokens"].apply(self.lemmatization)
        df["processed_text"] = df["tokens"].apply(lambda x: ' '.join(x))

        df = df[df["processed_text"] != "unknown"]

        if "label" not in df.columns:
            print("Labeling sentiment...")
            df["label"] = df["processed_text"].apply(self.sentiment_labeling)

        df.dropna(inplace=True)

        print("Balancing data with undersampling...")
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(df[["processed_text"]], df["label"])
        df_resampled = pd.DataFrame({"processed_text": X_resampled["processed_text"], "label": y_resampled})

        print(f"Saving preprocessed data to {output_path}...")
        df_resampled.to_csv(output_path, index=False)

        print("Preprocessing completed!")
        return df_resampled
    
if __name__ == "__main__":
    preprocessor = Preprocessor()
    df_preprocessed = preprocessor.preprocess("data/playstore.csv", "data/playstore_preprocessed.csv")