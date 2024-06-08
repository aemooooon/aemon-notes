---
title: 
draft: false
tags:
  - env
  - python
date: 2024-06-03
---


> [!aspiration]
> 最近发现很多Data Science的同学，学了半年480，还不知道如何正确的管理自己的Python Project，运行环境和版本控制工具。所以弄一个note放在这里！

# Python Project Setup

## Part 1: Setting Up the Project

1. **Create a Project Folder**
    ```bash
    mkdir my_python_project
    cd my_python_project
    ```

2. **Set Up a Virtual Environment**
    ```bash
    python -m venv venv
    ```

3. **Activate the Virtual Environment**
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4. **Create a Python Program**
    - Create a file called `main.py` in your project folder:
        ```python
        # main.py
        print("Hello, world!")
        ```

5. **Install Dependencies**
    - For this example, let's install `requests` as a dependency:
        ```bash
        python -m pip install requests
        ```

6. **Generate `requirements.txt`**
    ```bash
    # 把当前所用到的库都存到这个文件，方便新环境快速安装
    python -m pip freeze > requirements.txt 
    ```

For more information can check: https://docs.python.org/3/tutorial/venv.html

## Part 2: Adding Version Control

1. **Initialize a Git Repository**
    ```bash
    git init
    ```

2. **Create a `.gitignore` File**
    - Add the following lines to ignore the virtual environment and other unnecessary files:
        ```
        venv/
        __pycache__/
        *.pyc
        .DS_Store
        ```

3. **Commit Your Work Frequently**
    - It's essential to commit your work often. For example, if you have an assignment or project with 10 tasks, commit at least once after completing each task:
        ```bash
        git add .
        git commit -m "Completed task 1"
        ```

4. **Push to GitHub**
    - Create a new repository on GitHub and follow the instructions to add the remote repository:
        ```bash
        git remote add origin https://github.com/yourusername/my_python_project.git
        git push -u origin master
        ```

## Part 3: Running on New Environment

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/my_python_project.git
    cd my_python_project
    ```

2. **Set Up a Virtual Environment**
    ```bash
    python -m venv venv
    ```

3. **Activate the Virtual Environment**
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4. **Install Dependencies**
    ```bash
    python -m pip install -r requirements.txt
    ```

5. **Run the Program**
    ```bash
    python main.py
    ```

By following these steps, you can ensure that your Python project runs smoothly across different environments and computers.
