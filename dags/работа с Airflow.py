from airflow import DAG
# We need to import the operators used in our tasks
from airflow.operators.python import PythonOperator
# We then import the days_ago function
from airflow.utils.dates import days_ago
from datetime import timedelta

import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def gener_value(ind = 'С1'):
    from sklearn.datasets import make_blobs
    df=pd.read_csv('/workspaces/-/data_of_oilcompany_delt_1.csv', index_col='group')
    # df = df.drop(columns=['class.1'])
    centers = df.loc[df.index == ind, ['∆Выручки 2021 - Mean', '∆Выручки 2022 - Mean', '∆Чистой прибыли 2021 - Mean', '∆Чистой прибыли 2022	- Mean', '∆Активов 2021 - Mean', '∆Активов 2022 - Mean', '∆Собств. капитала 2021 - Mean', '∆Собств. капитала 2022 - Mean', '∆Рыночн. капитализации 2021 - Mean', '∆Рыночн. капитализации 2022 - Mean',	'∆Затрат 2021 - Mean', '∆Затрат 2022 - Mean']]
    cluster_std = df.loc[df.index == ind, ['∆Выручки 2021 - Variance', '∆Выручки 2022 - Variance', '∆Чистой прибыли 2021 - Variance', '∆Чистой прибыли 2022	- Variance', '∆Активов 2021 - Variance', '∆Активов 2022 - Variance', '∆Собств. капитала 2021 - Variance', '∆Собств. капитала 2022 - Variance', '∆Рыночн. капитализации 2021 - Variance', '∆Рыночн. капитализации 2022 - Variance',	'∆Затрат 2021 - Variance', '∆Затрат 2022 - Variance']]
    centers = centers.to_numpy()
    cluster_std = cluster_std.to_numpy()
    X, y = make_blobs(n_samples=200, centers=centers, cluster_std=cluster_std, shuffle=True, random_state=42)
    return X,y

def gener_group1():
    X,y = gener_value(ind = 'С1')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    df_train = pd.DataFrame(X_train, columns=['∆Выручки 2021', '∆Выручки 2022', '∆Чистой прибыли 2021', '∆Чистой прибыли 2022', '∆Активов 2021', '∆Активов 2022', '∆Собств. капитала 2021', '∆Собств. капитала 2022', '∆Рыночн. капитализации 2021', '∆Рыночн. капитализации 2022',	'∆Затрат 2021', '∆Затрат 2022'])
    df_train['target'] = y_train
    df_test = pd.DataFrame(X_test, columns=['∆Выручки 2021', '∆Выручки 2022', '∆Чистой прибыли 2021', '∆Чистой прибыли 2022', '∆Активов 2021', '∆Активов 2022', '∆Собств. капитала 2021', '∆Собств. капитала 2022', '∆Рыночн. капитализации 2021', '∆Рыночн. капитализации 2022',	'∆Затрат 2021', '∆Затрат 2022])
    df_test['target'] = y_test
    df_train.to_csv('/opt/airflow/files/gener/df_train_1.csv', index=False)
    df_test.to_csv('/opt/airflow/files/gener/df_test_1.csv', index=False)
    
def gener_group2():
    X,y = gener_value(ind = 'С2')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    df_train = pd.DataFrame(X_train, columns=['∆Выручки 2021', '∆Выручки 2022', '∆Чистой прибыли 2021', '∆Чистой прибыли 2022', '∆Активов 2021', '∆Активов 2022', '∆Собств. капитала 2021', '∆Собств. капитала 2022', '∆Рыночн. капитализации 2021', '∆Рыночн. капитализации 2022',	'∆Затрат 2021', '∆Затрат 2022])
    df_train['target'] = y_train
    df_test = pd.DataFrame(X_test, columns=['∆Выручки 2021', '∆Выручки 2022', '∆Чистой прибыли 2021', '∆Чистой прибыли 2022', '∆Активов 2021', '∆Активов 2022', '∆Собств. капитала 2021', '∆Собств. капитала 2022', '∆Рыночн. капитализации 2021', '∆Рыночн. капитализации 2022',	'∆Затрат 2021', '∆Затрат 2022])
    df_test['target'] = y_test
    df_train.to_csv('/opt/airflow/files/gener/df_train_2.csv', index=False)
    df_test.to_csv('/opt/airflow/files/gener/df_test_2.csv', index=False)
    
def gener_group3():
    X,y =gener_value(ind = 'С3')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    df_train = pd.DataFrame(X_train, columns=['∆Выручки 2021', '∆Выручки 2022', '∆Чистой прибыли 2021', '∆Чистой прибыли 2022', '∆Активов 2021', '∆Активов 2022', '∆Собств. капитала 2021', '∆Собств. капитала 2022', '∆Рыночн. капитализации 2021', '∆Рыночн. капитализации 2022',	'∆Затрат 2021', '∆Затрат 2022])
    df_train['target'] = y_train
    df_test = pd.DataFrame(X_test, columns=['∆Выручки 2021', '∆Выручки 2022', '∆Чистой прибыли 2021', '∆Чистой прибыли 2022', '∆Активов 2021', '∆Активов 2022', '∆Собств. капитала 2021', '∆Собств. капитала 2022', '∆Рыночн. капитализации 2021', '∆Рыночн. капитализации 2022',	'∆Затрат 2021', '∆Затрат 2022])
    df_test['target'] = y_test
    df_train.to_csv('/opt/airflow/files/gener/df_train_3.csv', index=False)
    df_test.to_csv('/opt/airflow/files/gener/df_test_3.csv', index=False)


def get_data_train():  #https://scikit-learn.org/stable/model_persistence.html
    # Your Python script goes here
    df1=pd.read_csv('/opt/airflow/files/gener/df_train_1.csv')
    df2=pd.read_csv('/opt/airflow/files/gener/df_train_2.csv')
    df3=pd.read_csv('/opt/airflow/files/gener/df_train_3.csv')
    df4 = pd.concat([df1, df2, df3])
    df4.to_csv('/opt/airflow/files/gener/df_train.csv', index=False)

def get_data_test():  #https://scikit-learn.org/stable/model_persistence.html
    # Your Python script goes here
    df1=pd.read_csv('/opt/airflow/files/gener/df_test_1.csv')
    df2=pd.read_csv('/opt/airflow/files/gener/df_test_2.csv')
    df3=pd.read_csv('/opt/airflow/files/gener/df_test_3.csv')
    df4 = pd.concat([df1, df2, df3])
    df4.to_csv('/opt/airflow/files/gener/df_test.csv', index=False)

    
    
def train01():
    load_train(data_set='/opt/airflow/files/gener/df_train.csv', dump_model='/opt/airflow/files/gener/filename.joblib')
    
def train02():
    load_train(data_set='/opt/airflow/files/gener/df_train.csv', dump_model='/opt/airflow/files/gener/filename.joblib')

def load_train(data_set='/opt/airflow/files/gener/df_train.csv', dump_model='/opt/airflow/files/gener/filename.joblib'): 
    # Your Python script goes here
    from sklearn import svm
    from joblib import dump, load
    clf = svm.SVC()
    df_train=pd.read_csv(data_set) #skipcols=usecols=[1,:]
    X_train=df_train.iloc[:,0:3].to_numpy()
    y_train=df_train.iloc[:,4].to_numpy()
    clf.fit(X_train, y_train)
    dump(clf, dump_model) 

    
def load_test():
    import pandas as pd
    import numpy as np
    from sklearn import svm
    from joblib import dump, load
    clf2 = load('/opt/airflow/files/gener/filename.joblib')
    df_test=pd.read_csv('/opt/airflow/files/gener/df_test.csv') 
    X_test=df_test.iloc[:,0:3].to_numpy()
    y_test=df_test.iloc[:,4].to_numpy()
    predict = clf2.predict(X_test)
    df = pd.DataFrame(predict)
    df.to_csv('/opt/airflow/files/gener/predict.csv', index=False)

    

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'email': ['airflow@my_first_dag.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}


paral = DAG(
    'paral',
    default_args=default_args,
    description='Our DAG 2',
    schedule_interval=timedelta(days=1),
)

gener_group1 = PythonOperator(
    task_id='gener_class1',
    python_callable=gener_group1, #написать функцию
    dag=paral,
)
gener_group2 = PythonOperator(
    task_id='gener_class2',
    python_callable=gener_group2, #написать функцию
    dag=paral,
)
gener_group3 = PythonOperator(
    task_id='gener_class3',
    python_callable=gener_group3, #написать функцию
    dag=paral,
)

get_data_train = PythonOperator(
    task_id='get_data_train',
    python_callable=get_data_train, #написать функцию
    dag=paral,
)
get_data_test = PythonOperator(
    task_id='get_data_test',
    python_callable=get_data_test, #написать функцию
    dag=paral,
)

fit_model01 = PythonOperator(
    task_id='fit_model01',
    python_callable=train01, #написать функцию
    dag=paral,
)
fit_model02 = PythonOperator(
    task_id='fit_model02',
    python_callable=train02, #написать функцию
    dag=paral,
)

test_model = PythonOperator(
    task_id='test_model',
    python_callable=load_test, #написать функцию
    dag=paral,
)


