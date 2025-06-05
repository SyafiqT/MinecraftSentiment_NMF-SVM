from flask import render_template
from utils import data_loader, visualization, metrics

def register_routes(app):
    data_loader.load_all_data()  # Load data at startup
    visualization.generate_plots()  # Generate plots once

    @app.route('/')
    def landing():
        return render_template('index.html')

    @app.route('/scrape')
    def scraped():
        return render_template('scrape.html', data=data_loader.scraped_data.head(1000))

    @app.route('/preprocess')
    def preprocess():
        return render_template('preprocess.html', data=data_loader.preprocessed_data.head(1000))

    @app.route('/visual')
    def visual():
        topics = visualization.get_topics()
        label_distribution = data_loader.preprocessed_data['label'].value_counts().to_dict()
        topic_distribution = data_loader.topic_distribution
        classification_metrics = metrics.get_classification_metrics()
        example_predictions = metrics.get_example_predictions()

        return render_template(
            'visual.html',
            topics=topics,
            topic_distribution=topic_distribution,
            label_distribution=label_distribution,
            metrics=classification_metrics,
            example_predictions=example_predictions
        )
