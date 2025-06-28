from src.config.config_loader import load_config
from src.data.data_loader import load_data
from src.preprocess.preprocess import preprocess_data
from src.train.model_trainer import train_model
from src.evaluate.model_evaluator import evaluate_model
import mlflow

if __name__ == "__main__":
    config = load_config()
    mlflow.set_tracking_uri(config['mlflow_config']['tracking_uri'])
    mlflow.set_experiment(config['mlflow_config']['experiment_name'])

    df = load_data(config['data_source']['path'])
    X_train, X_test, y_train, y_test = preprocess_data(df, config['target_column'])

    model = train_model(X_train, y_train, config)
    score = evaluate_model(model, X_test, y_test)
    print("Accuracy:", score)
