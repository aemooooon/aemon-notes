---
title: 
draft: false
tags:
  - aws
  - ec2
date: 2024-06-03
---


1. Create an `airflow` folder and then get into it
2. Create a [Python3 virtual environment](Python%20Virtual%20Environment.md) in it
3. Run the installation command in PyPI way and ensure the Python version is the same. For example, `pip install 'apache-airflow==2.9.1' --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.1/constraints-3.12.txt"`
4. Set up `export AIRFLOW_HOME=/home/ubuntu/airflow`
5. Initialisation `airflow db init`
6. Run Web server `airflow webserver -p 8080`
7. Set up user `airflow users create --username admin --firstname Aemon --lastname Wang --role Admin --email aemooooon@gmail.com` When first time access the link might face some error, try `airflow db init` again or `airflow db migrate`
8. Run scheduler service `airflow scheduler`
## 设置默认数据库为PSQL

 准备好数据库，然后修改文件，把默认的sqlite换成你的psql数据库就行了。
```
airflow.cfg

sql_alchemy_conn = postgresql+psycopg2://airflowuser:yourpassword@airflow-db.xxxxxxxx.us-west-2.rds.amazonaws.com:5432/yourdbname

executor = SequentialExecutor 
executor = SequentialExecutor

load_examples = False
```
## 设置开发数据库为PSQL

>直接在airflow web 管理中心的connections管理里面添加连接字符串即可。

`lsof -i :8793`
