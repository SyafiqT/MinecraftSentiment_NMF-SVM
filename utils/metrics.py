import pandas as pd
from sklearn.metrics import classification_report
from utils import data_loader

def get_classification_metrics():
    results = data_loader.classification_results
    metrics = {}
    if 'actual_label' in results.columns and 'predicted_label' in results.columns:
        metrics['accuracy'] = (results['actual_label'] == results['predicted_label']).mean()
        class_report = classification_report(
            results['actual_label'],
            results['predicted_label'],
            output_dict=True
        )
        metrics['class_metrics'] = {
            k: {
                'precision': v['precision'],
                'recall': v['recall'],
                'f1': v['f1-score'],
                'support': v['support']
            }
            for k, v in class_report.items()
            if k not in ['accuracy', 'macro avg', 'weighted avg']
        }
    else:
        metrics['accuracy'] = 0
        metrics['class_metrics'] = {}
    return metrics

def get_example_predictions(n=10):
    results = data_loader.classification_results
    if len(results) > 0 and 'is_correct' in results.columns:
        results['is_correct'] = results['is_correct'].astype(bool)
        correct = results[results['is_correct']].sample(n=n//2) if sum(results['is_correct']) > 0 else pd.DataFrame()
        incorrect = results[~results['is_correct']].sample(n=n//2) if sum(~results['is_correct']) > 0 else pd.DataFrame()
        examples = pd.concat([correct, incorrect])

        return [
            {
                'text': row['text'][:100] + '...' if len(row['text']) > 100 else row['text'],
                'actual': row['actual_label'],
                'predicted': row['predicted_label'],
                'is_correct': row['is_correct']
            }
            for _, row in examples.iterrows()
        ]
    return []
