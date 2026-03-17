

Prequities
Install docker

Python3 installed

Getting started

1. Create python virtual environment (venv) at root of this repo

```cmd
python2 -m venv venv

<!-- activate virtual environement -->
source venv/bin/activate
```

2. Install python dependencies

```cmd 
pip install -r requirements 
```

3. Run server

```cmd
python main.py
```



Run text embeddings api

docker build -t embeddings-api services/text-embeddings-api/ 
docker run -p 11434:11434 embeddings-api


Run vector db

docker build -t vector-db services/vector-db/
docker run -p 6333:6333 vector-db

