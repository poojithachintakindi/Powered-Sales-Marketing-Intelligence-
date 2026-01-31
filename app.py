import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
from utils.analytics import compute_basic_analytics
from utils.model import train_conversion_model, predict_conversion_probabilities
from utils.generative import generate_ai_insights

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key'  # Replace in production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard', methods=['POST'])
def dashboard():
    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file.')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            flash(f'Failed to read CSV: {e}')
            return redirect(url_for('index'))

        # Preprocess and compute analytics
        try:
            analytics, processed_df, schema = compute_basic_analytics(df)
        except Exception as e:
            flash(f'Analytics error: {e}')
            return redirect(url_for('index'))

        # Train model and predict conversion probabilities
        model_info = None
        predictions = None
        try:
            model, X_columns, metrics = train_conversion_model(processed_df, schema)
            preds = predict_conversion_probabilities(model, processed_df, X_columns)
            model_info = {
                'algorithm': 'Logistic Regression',
                'metrics': metrics,
                'feature_count': len(X_columns)
            }
            predictions = preds
        except Exception as e:
            # Model is optional if data lacks target or suitable features
            model_info = {
                'algorithm': 'Logistic Regression',
                'metrics': {'note': f'Model not trained: {e}'}
            }

        # Generate AI insights using placeholder LLM call
        try:
            ai_insights = generate_ai_insights(analytics, model_info)
        except Exception as e:
            ai_insights = [
                'AI insights generation failed. Showing fallback insights.',
                f'Error: {e}'
            ]

        # Prepare compact tables for display
        top_campaigns_table = analytics.get('top_campaigns_table')
        predictions_table = None
        if predictions is not None:
            show_cols = []
            # Keep likely relevant columns for display
            for c in ['customer_id', 'campaign', 'prob_conversion']:
                if c in predictions.columns:
                    show_cols.append(c)
            if not show_cols:
                # default to first few columns
                show_cols = list(predictions.columns[:3])
            predictions_table = predictions[show_cols].head(20).to_dict(orient='records')

        return render_template(
            'dashboard.html',
            analytics=analytics,
            model_info=model_info,
            ai_insights=ai_insights,
            top_campaigns_table=top_campaigns_table,
            predictions_table=predictions_table
        )

    flash('Unsupported file type. Please upload a CSV file.')
    return redirect(url_for('index'))


if __name__ == '__main__':
    # For local development
    app.run(host='127.0.0.1', port=5000, debug=True)
