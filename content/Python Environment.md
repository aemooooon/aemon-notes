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

# 什么是Pipenv

Pipenv 是一种用于Python项目的依赖管理和虚拟环境管理工具。它集成了`pip`和`virtualenv`的功能，简化了项目设置，确保依赖的一致性和安全性。

# 安装Pipenv

首先，你需要在系统中安装Pipenv。你可以使用以下命令来安装它：

```bash
pip install pipenv
```

# 初始化项目

在你的项目目录中运行以下命令以初始化Pipfile：

```bash
pipenv install
```

这将创建一个`Pipfile`，用于管理项目的依赖。

# 安装依赖

使用Pipenv安装依赖非常简单。以下命令将在你的`Pipfile`中添加`requests`库：

```bash
pipenv install requests
```

# 管理开发依赖

如果你需要安装仅在开发过程中使用的依赖，例如测试框架，可以使用`--dev`标志：

```bash
pipenv install --dev pytest
```

这将把`pytest`添加到`Pipfile`中的`[dev-packages]`部分。

# 使用虚拟环境

Pipenv会自动创建和管理虚拟环境。你可以使用以下命令激活虚拟环境：

```bash
pipenv shell
```

在虚拟环境中，你可以正常运行Python命令和脚本。要退出虚拟环境，只需运行：

```sh
exit
```

# 生成和使用锁定文件

Pipenv会自动生成`Pipfile.lock`文件，记录确切的包版本，以确保环境的一致性。要安装`Pipfile.lock`中的依赖，可以使用：

```bash
pipenv install
```

# 更新依赖

你可以使用以下命令更新所有依赖到最新版本，并更新`Pipfile.lock`：

```bash
pipenv update
```

# 卸载依赖

要卸载某个依赖，你可以使用以下命令：

```bash
pipenv uninstall <package_name>
```

# 检查依赖安全

Pipenv可以检查依赖中是否存在已知的安全漏洞。使用以下命令进行检查：

```bash
pipenv check
```

# 实际开发示例

假设你有一个名为`my_project`的项目。以下是使用Pipenv管理项目依赖和虚拟环境的示例：

```sh
# 创建项目目录
mkdir my_project
cd my_project

# 初始化Pipenv环境
pipenv install

# 安装生产依赖
pipenv install requests

# 安装开发依赖
pipenv install --dev pytest

# 激活虚拟环境
pipenv shell

# 在虚拟环境中运行Python脚本
python script.py

# 退出虚拟环境
exit
```
