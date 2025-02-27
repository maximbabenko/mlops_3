import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from datetime import datetime

# Проверка и создание папки для сохранения артефактов
os.makedirs("public", exist_ok=True)

# Загрузка данных
data_path = "data/iris.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Файл данных не найден: {data_path}")

data = pd.read_csv(data_path)

# Приведение всех целочисленных столбцов к float
data = data.astype({col: float for col in data.select_dtypes(include=["int"]).columns})

# Разделение признаков и целевой переменной
X = data.drop(columns=["Species"])
y = data["Species"]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Настройка MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris_classification_experiment")

def save_confusion_matrix(y_true, y_pred, model_name):
    """Создание и сохранение матрицы ошибок."""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join("public", filename))
    plt.close()
    return filename

def log_model_with_metrics(model, model_name, X_train, X_test, y_train, y_test):
    """Логирование модели и метрик в MLflow."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"{model_name}_{timestamp}"):
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Метрики
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
            }

            # Логирование параметров и метрик
            mlflow.log_param("model", model_name)
            for param_name, param_value in model.get_params().items():
                mlflow.log_param(param_name, param_value)

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Сохранение матрицы ошибок и отчета о классификации
            cm_filename = save_confusion_matrix(y_test, y_pred, model_name)
            mlflow.log_artifact(os.path.join("public", cm_filename))

            report_filename = f"classification_report_{model_name.lower().replace(' ', '_')}.txt"
            with open(os.path.join("public", report_filename), "w") as f:
                f.write(classification_report(y_test, y_pred, zero_division=0))
            mlflow.log_artifact(os.path.join("public", report_filename))

            # Сохранение модели в MLflow
            mlflow.sklearn.log_model(model, "model", input_example=X_test.iloc[:1], registered_model_name=model_name)

            return {
                "run_id": mlflow.active_run().info.run_id,
                "confusion_matrix_path": cm_filename,
                "classification_report_path": report_filename,
            }
        except Exception as e:
            print(f"Ошибка в процессе логирования модели {model_name}: {e}")
            raise

def run_experiments():
    """Запуск нескольких экспериментов с разными параметрами."""
    experiments = [
        {'model': LogisticRegression(max_iter=1000, random_state=42), 'name': 'Logistic Regression (C=1.0)'},
        {'model': LogisticRegression(max_iter=2000, random_state=42, C=0.5), 'name': 'Logistic Regression (C=0.5)'},
        {'model': DecisionTreeClassifier(max_depth=10, random_state=42), 'name': 'Decision Tree (max_depth=10)'},
        {'model': DecisionTreeClassifier(max_depth=15, random_state=42), 'name': 'Decision Tree (max_depth=15)'}
    ]

    results = []
    for exp in experiments:
        print(f"Starting experiment with {exp['name']}...")
        try:
            run_results = log_model_with_metrics(exp['model'], exp['name'], X_train, X_test, y_train, y_test)
            run_id = run_results["run_id"]
            run_data = mlflow.get_run(run_id).data.metrics
            results.append({
                'RunID': run_id,
                'Model': exp['name'],
                'accuracy': run_data['accuracy'],
                'precision': run_data['precision'],
                'recall': run_data['recall'],
                'f1_score': run_data['f1_score'],
                'confusion_matrix_path': run_results["confusion_matrix_path"],
                'classification_report_path': run_results["classification_report_path"],
            })
            print(f"Completed experiment with {exp['name']}")
        except Exception as e:
            print(f"Ошибка при выполнении эксперимента {exp['name']}: {e}")

    return pd.DataFrame(results)

# Выполнение экспериментов
if __name__ == "__main__":
    try:
        results_df = run_experiments()
        print("Все эксперименты завершены!")
        print(results_df)
    except ModuleNotFoundError:
        print("Ошибка: MLflow не установлен или недоступен.")
        print("Убедитесь, что библиотека установлена, используя 'pip install mlflow'.")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
