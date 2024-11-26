import re
import nltk
import pickle
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class Preprocessing(BaseEstimator, TransformerMixin):
    def __init_(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:

        preprocessed_text = X.apply(self.preprocess)
        return pd.DataFrame(preprocessed_text, columns=['text'])

    def preprocess(self, text):

        text = text.lower() # lowercasing
        text = re.sub(r'https?://\S+|www\.\S+|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' ', text) # remove links
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text) # remove special chars
        text = nltk.word_tokenize(text) #tokenization

        st_word = set(stopwords.words('english')) #stop word removal
        text = [i for i in text if i not in st_word]

        ps = PorterStemmer()
        text = [ps.stem(i) for i in text]

        return ' '.join(text)
    

class Model:
    def __init__(self):
        pass

    def main(self, train_data_path, save_model_path: Path) -> None:
        # fetch data
        df = pd.read_csv(train_data_path)

        X = df['text']
        y = df['target']
        
        del df

        coluumn_transformer = ColumnTransformer([
        ('Vectorize', TfidfVectorizer(max_features=5000), 'text')
        ])

        voting_classifier = VotingClassifier([
            ('MultinomialNB', MultinomialNB()),
            ('LogisticRegression', LogisticRegression())
        ])


        final_pipeline = Pipeline(steps=[
            ('Text Preprocessin', Preprocessing()),
            ('Vectorizer', coluumn_transformer),
            ('Model', voting_classifier)
        ])

        final_pipeline.fit(X, y)
        # svame model
        with open(save_model_path, 'wb') as f:
            pickle.dump(final_pipeline, f)