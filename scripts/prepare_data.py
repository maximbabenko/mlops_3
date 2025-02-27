import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(input_path):
    # Загрузить данные
    df = pd.read_csv(input_path)
    return df

def clean_data(df):
    # Удаляем пропущенные значения, если есть
    df = df.dropna()
    return df

def scale_features(df):
    # Стандартизация признаков (кроме колонки 'species')
    scaler = StandardScaler()
    feature_columns = df.columns[:-1]  # Применяется ко всем столбцам, кроме последнего
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

def save_data(df, output_path):
    # Сохранить обработанные данные
    df.to_csv(output_path, index=False)

def main(input_path, output_path):
    # Основной процесс обработки данных
    df = load_data(input_path)  # Инициализация df
    df = clean_data(df)         # Применение очистки данных
    df = scale_features(df)     # Применение стандартизации
    save_data(df, output_path)  # Сохранение обработанных данных

if __name__ == "__main__":
    # Убедитесь, что пути к данным корректны
    input_path = "data/iris.csv"
    output_path = "data/iris_processed.csv"
    main(input_path, output_path)

