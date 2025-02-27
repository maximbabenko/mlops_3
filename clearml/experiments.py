import joblib
from clearml import Task
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Проверка и создание папки для сохранения артефактов
os.makedirs("public", exist_ok=True)

# Загрузка данных
data_path = "data/iris.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Файл данных не найден: {data_path}")

data = pd.read_csv(data_path)

# Приведение всех целочисленных столбцов к float
data[data.select_dtypes(include=["int"]).columns] = data.select_dtypes(include=["int"]).astype(float)

# Разделение признаков и целевой переменной
X = data.drop(columns=["Species"])
y = data["Species"]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def save_confusion_matrix(y_true, y_pred, model_name):
    """Создание и сохранение матрицы ошибок."""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Сохранение матрицы ошибок
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    save_path = os.path.join("public", filename)
    plt.savefig(save_path)
    plt.close()
    return filename

def log_model_with_metrics(model, model_name, X_train, X_test, y_train, y_test):
    """Логирование модели и метрик в ClearML."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Инициализация задачи ClearML
    task = Task.init(project_name='iris_classification', task_name=f"{model_name}_{timestamp}", task_type=Task.TaskTypes.optimizer)
    
    try:
        # Обучение модели
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Логирование метрик
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }
        
        task.connect(metrics)

        # Логирование параметров модели
        task.set_parameters(model.get_params())
        task.set_parameters({"model": model_name})

        # Сохранение матрицы ошибок
        cm_filename = save_confusion_matrix(y_test, y_pred, model_name)
        task.upload_artifact(name=cm_filename, artifact_object=os.path.join("public", cm_filename))

        # Сохранение отчета о классификации
        report_filename = f"classification_report_{model_name.lower().replace(' ', '_')}.txt"
        report_save_path = os.path.join("public", report_filename)

        with open(report_save_path, "w") as f:
            f.write(classification_report(y_test, y_pred, zero_division=0))

        task.upload_artifact(name=report_filename, artifact_object=report_save_path)

        # Сохранение модели в ClearML
        model_filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, model_filename)
        task.upload_artifact(name=model_filename, artifact_object=model_filename)

        return {
            "task_id": task.id,
            "confusion_matrix_path": cm_filename,
            "classification_report_path": report_filename,
            "model_filename": model_filename,
        }
    except Exception as e:
        print(f"Ошибка в процессе логирования модели {model_name}: {e}")
        raise
    finally:
        task.close()  # Закрытие задачи в любом случае

def run_experiments():
    """Запуск нескольких экспериментов с разными параметрами."""
    experiments = [
        {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'name': 'Logistic Regression (C=1.0)'
        },
        {
            'model': LogisticRegression(max_iter=2000, random_state=42, C=0.5),
            'name': 'Logistic Regression (C=0.5)'
        },
        {
            'model': DecisionTreeClassifier(max_depth=10, random_state=42),
            'name': 'Decision Tree (max_depth=10)'
        },
        {
            'model': DecisionTreeClassifier(max_depth=15, random_state=42),
            'name': 'Decision Tree (max_depth=15)'
        }
    ]

    results = []
    for exp in experiments:
        print(f"Starting experiment with {exp['name']}...")
        try:
            run_results = log_model_with_metrics(exp['model'], exp['name'], X_train, X_test, y_train, y_test)
            results.append({
                'TaskID': run_results["task_id"],
                'Model': exp['name'],
                'confusion_matrix_path': run_results["confusion_matrix_path"],
                'classification_report_path': run_results["classification_report_path"],
                'model_filename': run_results["model_filename"],
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
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
