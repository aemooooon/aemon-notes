https://docs.python.org/3/tutorial/venv.html

```python
python -m venv env-name
windows: tutorial-env\Scripts\activate
MacOS & Linux: source tutorial-env/bin/activate
deactivate

python -m pip install novas
python -m pip install requests==2.6.0
python -m pip install --upgrade requests
python -m pip uninstall requests
python -m pip show requests
python -m pip list
python -m pip freeze > requirements.txt
python -m pip install -r requirements.txt

python3 -m pip install --upgrade pip
```