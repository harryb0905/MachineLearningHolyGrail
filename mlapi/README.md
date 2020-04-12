## MLAPI

This repository contains code for an API built using Flask RESTful to serve machine learning models.

First, run the virtual environment:
```
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Then run the API:
```python
python app.py
```

Send a GET request to obtain a list of currently supported models:

```
curl 127.0.0.1:5000/models
```

```
[
    "LogisticRegression",
    "SVC",
    "RandomForest"
]
```