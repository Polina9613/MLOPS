import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def download_data():
    """Загрузка данных"""
    df = pd.read_csv('insurance_miptstats.csv', sep=',')
    df.to_csv("insurance_raw.csv", index=False)
    return df

def clear_data(path2df):
    """Очистка и предобработка данных"""
    df = pd.read_csv(path2df)
    
    # Создание признака возраста
    df['age'] = pd.to_datetime('today').year - pd.to_datetime(df['birthday']).dt.year
    df = df.drop(columns=['birthday'])
    
    # Удаление выбросов
    df = df[(df['age'] >= 18) & (df['age'] <= 80)]
    df = df[(df['bmi'] >= 15) & (df['bmi'] <= 50)]
    df = df[(df['charges'] >= 1000) & (df['charges'] <= 50000)]
    
    # Кодирование категориальных признаков
    cat_columns = ['sex', 'smoker', 'region']
    num_columns = ['age', 'bmi', 'children']
    
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    df_ordinal = pd.DataFrame(ordinal.transform(df[cat_columns]), columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    
    # Сохранение очищенных данных
    df.to_csv('insurance_clean.csv', index=False)
    return True

if __name__ == "__main__":
    download_data()
    clear_data("insurance_raw.csv")
