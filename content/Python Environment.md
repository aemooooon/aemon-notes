---
title: 
draft: false
tags:
  - env
  - python
date: 2024-06-03
---
> [!aspiration]
> 在macOS M1芯片的电脑上，常常会遇到安装了多个版本的Python的问题。这些版本可能通过不同的途径安装，如Homebrew、Anaconda、以及直接从Python官网下载安装包。由于官方已经提供了`venv`模块来创建虚拟环境，因此为了简化和方便管理，决定使用Homebrew来安装和更新Python。本文将介绍如何查看和删除其他版本的Python，如何使用Homebrew安装和更新Python版本，并如何配置系统默认的Python环境。

# 查看当前系统中的Python版本

使用以下命令可以查看系统中所有的Python 3版本路径：

```sh
which -a python3
```

# 删除其他安装的Python版本

## 删除Anaconda安装的Python

首先，删除Anaconda安装目录：

```sh
rm -rf ~/anaconda3
```

然后删除Anaconda的配置文件和缓存目录：

```sh
rm -rf ~/.condarc ~/.conda ~/.continuum
```

## 删除官方安装的Python版本

通过官方安装包安装的Python版本通常位于`/Library/Frameworks/Python.framework/Versions/`下，可以删除该目录下不需要的版本：

```sh
sudo rm -rf /Library/Frameworks/Python.framework/Versions/3.x
```

# 使用Homebrew安装和更新Python

## 安装Homebrew

如果尚未安装Homebrew，可以通过以下命令安装：

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## 安装Python

通过Homebrew安装Python：

```sh
brew install python
```

## 更新Python版本

更新Homebrew和已安装的Python版本：

```sh
brew update
brew upgrade python
```

# 配置系统默认Python环境

#### 修改`.zshrc`文件

编辑`~/.zshrc`文件，添加以下内容以将Homebrew安装的Python设置为默认的Python版本：

```sh
# Homebrew installed Python
export PATH="/opt/homebrew/bin:$PATH"
alias python=/opt/homebrew/bin/python3
alias python3=/opt/homebrew/bin/python3
```

对于Intel Mac，请使用：

```sh
# Homebrew installed Python
export PATH="/usr/local/bin:$PATH"
alias python=/usr/local/bin/python3
alias python3=/usr/local/bin/python3
```

## 重新加载`.zshrc`文件

使更改生效，重新加载`.zshrc`文件：

```sh
source ~/.zshrc
```

# 创建符号链接，将pip3链接到pip

创建符号链接，使`pip`指向`pip3`：

```sh
ln -s /opt/homebrew/bin/pip3 /opt/homebrew/bin/pip
```

对于Intel Mac：

```sh
ln -s /usr/local/bin/pip3 /usr/local/bin/pip
```

# 检查Python和pip版本

验证`python`和`pip`是否指向正确的版本：

```sh
which python
python --version

which pip
pip --version
```

# 更新pip

使用以下命令更新pip到最新版本：

```sh
pip install --upgrade pip

# some times might be need option due to OS policy
# pip install --upgrade pip --break-system-packages
```

# Python Virtual Environment

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
