import pandas as pd
import mlflow
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from mlflow.models import infer_signature

def eval_metrics(actual, pred):
    """Вычисление метрик качества"""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def prepare_data():
    """Подготовка данных для обучения"""
    df = pd.read_csv("./insurance_clean.csv")
    
    categorical_features = ['sex', 'smoker', 'region']
    numerical_features = ['age', 'bmi', 'children']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    return X, y, preprocessor

if __name__ == "__main__":
    X, y, preprocessor = prepare_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42)
    
    # Параметры для GridSearch
    params = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    }
    
    mlflow.set_experiment("Insurance_GradientBoosting")
    with mlflow.start_run():
        # Создание пайплайна
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(random_state=42))
        ])
        
        # Поиск лучших параметров
        clf = GridSearchCV(pipeline, params, cv=3, n_jobs=4, 
                          scoring='neg_mean_squared_error')
        clf.fit(X_train, y_train)
        
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        
        # Вычисление метрик
        rmse, mae, r2 = eval_metrics(y_val, y_pred)
        
        # Логирование параметров
        mlflow.log_params(clf.best_params_)
        mlflow.log_metrics({
            "rmse": rmse,
            "r2": r2,
            "mae": mae
        })
        
        # Логирование модели
        signature = infer_signature(X_train, best.predict(X_train))
        mlflow.sklearn.log_model(best, "model", signature=signature)
        
        # Сохранение модели
        with open("gb_insurance.pkl", "wb") as file:
            joblib.dump(best, file)
    
    # Получение пути к лучшей модели
    dfruns = mlflow.search_runs()
    path2model = dfruns.sort_values("metrics.r2", ascending=False).iloc[0]['artifact_uri'].replace("file://","") + '/model'
    print(path2model)
