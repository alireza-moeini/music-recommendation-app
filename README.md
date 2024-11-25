# Music Recommendation API

FastAPI + PyTorch music recommender with model saving and Docker support.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Docker Deployment](#docker-deployment)


## Project Structure

```
.
│
├── main.py
├── model.py
├── trainer.py
├── data.py
├── requirements.txt
├── README.md
├── .gitignore
├── Dockerfile
├── docker-compose.yml
│
├── app/
│   ├── __init__.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── recommend.py
│   └── utils/
│       ├── __init__.py
│       └── mappings.py
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── models/
│   ├── __init__.py
│   └── model_store/
│
├── data_files/
│   ├── raw/
│   └── processed/
│
└── scripts/
    ├── train_local.py
    └── eval.py
```


## Installation

1. Clone the repository:

```bash
git clone https://github.com/alireza-moeini/music-recommendation-app.git
cd music-recommendation-app
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Dataset: the dataset is generated automatically in [data.py](./data.py). However, custom datasets may be used.

## Usage
1. Run the server:

```bash
python main.py
```

2. Train the model

```
POST http://localhost:8000/train
Body: {"epochs": 3}
```

3. Use the trained model to get recommendations:

```
GET http://localhost:8000/recommend/5?k=10
```

## Docker Deployment

1. Build the Docker image:

```bash
docker build -t music-recommendation-app .
```

2. Run the Docker container:

```bash
docker run -d -p 8000:8000 music-recommendation-app
```