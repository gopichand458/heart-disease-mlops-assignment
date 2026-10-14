# Heart Disease Risk Prediction (MLOps Assignment 1)

A classifier that predicts heart disease from the UCI Heart Disease (Cleveland)
data, served as an API in Docker and Kubernetes, with CI and basic monitoring.

## Overall flow

    UCI data -> download + clean -> train (MLflow) -> models/model.joblib
                                                          |
    GitHub push -> GitHub Actions (lint, train, test, docker build)
                                                          |
    Docker image -> Kubernetes (2 pods + service) -> Prometheus -> Grafana

## Folder layout

    data/            raw and cleaned csv
    notebooks/       01_eda.ipynb
    src/             download_data, clean_data, features, train, api
    tests/           pytest tests
    models/          saved model
    artifacts/       roc curves and confusion matrices
    k8s/             deployment, service, monitoring manifests
    monitoring/      exported Grafana dashboard
    screenshots/     evidence for the report
    .github/workflows/ci.yml

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-dev.txt

## Get and clean the data

    python -m src.download_data
    python -m src.clean_data

If the download fails behind a proxy, download the zip from the UCI page by hand
and save it as data/raw/heart_disease.zip, then run the script again.

## Train

    python -m src.train

Trains logistic regression and random forest with 5-fold cross validation,
logs runs to MLflow and saves the best model to models/model.joblib.
To look at the runs:

    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

## Test and lint

    pytest -q
    flake8 src tests --max-line-length=100

## Run the API locally

    uvicorn src.api:app --port 8000

    curl -s localhost:8000/health
    curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"age":63,"sex":1,"cp":4,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":1,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'

Endpoints: /health, /predict, /metrics/

## Docker

    docker build -t heart-api:1.0 .
    docker run --rm -p 8000:8000 heart-api:1.0

## Kubernetes (Docker Desktop cluster)

The cluster runs inside a container, so the image has to be loaded into it:

    docker save heart-api:1.0 | docker exec -i desktop-control-plane ctr -n k8s.io images import -
    kubectl apply -f k8s/
    kubectl get pods,svc
    kubectl port-forward svc/heart-api 8080:80

Then use localhost:8080 instead of 8000. On a Mac the load balancer IP is not
reachable from the host, so the port-forward is how I reach it.

## Monitoring

k8s/monitoring.yaml starts Prometheus and Grafana.

    kubectl port-forward svc/prometheus 9090:9090
    kubectl port-forward svc/grafana 3000:3000

Grafana login is admin / admin. Add Prometheus as a data source with the URL
http://prometheus:9090. The dashboard is exported in monitoring/grafana_dashboard.json.

## Results

| Model | CV ROC-AUC | Test accuracy | Test recall | Test ROC-AUC |
|---|---|---|---|---|
| Logistic regression | 0.901 | 0.885 | 0.929 | 0.968 |
| Random forest | 0.900 | 0.852 | 0.857 | 0.956 |

Logistic regression was kept because it is simpler and has the better recall.

## Notes

- MLflow 3 saves sklearn models with skops and refused the numpy dtype and the
  sklearn tree type until they were passed as trusted types in train.py.
- Missing values are filled inside the sklearn pipeline, not before the split,
  so the test data does not leak into the fill values.
- I added one feature, hr_reserve, which is (220 - age) - thalach.
