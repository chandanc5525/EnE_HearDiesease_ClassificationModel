from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def train_model(X_train, y_train, config):
    model_type = config['model_params']['model_type']
    params = config['model_params']['params']

    if model_type == "RandomForest":
        model = RandomForestClassifier(**params)
    else:
        raise ValueError("Unsupported model")

    mlflow.start_run()
    mlflow.log_params(params)

    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")

    mlflow.end_run()
    return model
