import pickle
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score
import json

class Evaluation:
    def __init__(self):
        pass

    def main(self, model_path: Path, test_data_path: Path) -> None:
        # load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # fetch data
        df = pd.read_csv(test_data_path)

        X = df['text']
        y = df['target']

        del df

        y_pred = model.predict(X)

        # save score in scores.jsos, acccuracy, precision, f1
        scores = {
            'accuracy': accuracy_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'roc_auc_score': roc_auc_score(y, y_pred)
        }
        # save scores
        with open(Path('scores.json'), 'w') as f:
            json.dump(scores, f)
