"""Azure Machine Learning helper: train in the cloud, register and fetch the model.

Requires `pip install -r requirements-azure.txt`, `az login`, and the AZURE_* values in .env.

Usage:
  python azure_ml.py submit-job [--wait]          # run train_model.py on the Azure ML compute
  python azure_ml.py status <job> [--wait]        # check (or wait for) a submitted job
  python azure_ml.py register                     # register the local weather_model.pkl
  python azure_ml.py register --from-job <job>    # register the model produced by a cloud job
  python azure_ml.py download [--version N]       # pull a registered model for the edge module
  python azure_ml.py list                         # list registered versions
"""
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

from config import (
    AZURE_ML_COMPUTE, AZURE_ML_WORKSPACE, AZURE_MODEL_NAME, AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID, BASE_DIR, MODEL_PATH,
)

JOB_FILES = ["train_model.py", "weather_data.csv"]
BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
TERMINAL_STATES = {"completed", "failed", "canceled", "notresponding"}


def status_of(job):
    # Newer SDKs return an enum (JobStatus.COMPLETED), older ones a string ("Completed")
    status = getattr(job.status, "value", job.status)
    return str(status).split(".")[-1].lower()


def ml_client():
    missing = [name for name, value in {
        "AZURE_SUBSCRIPTION_ID": AZURE_SUBSCRIPTION_ID,
        "AZURE_RESOURCE_GROUP": AZURE_RESOURCE_GROUP,
        "AZURE_ML_WORKSPACE": AZURE_ML_WORKSPACE,
    }.items() if not value]
    if missing:
        raise SystemExit(f"Set {', '.join(missing)} in .env first")
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    return MLClient(DefaultAzureCredential(), AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE)


def submit_job(args):
    from azure.ai.ml import command
    from azure.ai.ml.entities import Environment

    ml = ml_client()
    with tempfile.TemporaryDirectory() as src:  # upload only what the job needs
        for name in JOB_FILES:
            shutil.copy(BASE_DIR / name, src)
        job = command(
            code=src,
            command="python train_model.py",
            environment=Environment(
                name="weather-ml-environment",
                conda_file=str(BASE_DIR / "conda.yml"),
                image=BASE_IMAGE,
            ),
            compute=AZURE_ML_COMPUTE,
            experiment_name="weather-edge-ai",
            display_name="weather-rain-model-training",
        )
        job = ml.jobs.create_or_update(job)
    print(f"Submitted job: {job.name}")
    print(f"Studio URL:    {job.studio_url}")
    if args.wait:
        wait_for_job(ml, job.name)


def wait_for_job(ml, name):
    # Polls the status instead of ml.jobs.stream(), which fails on workspaces whose
    # storage uses identity-based access ("binascii.Error: Incorrect padding").
    last = None
    while True:
        status = status_of(ml.jobs.get(name))
        if status != last:
            print(f"[{time.strftime('%H:%M:%S')}] {name}: {status}")
            last = status
        if status in TERMINAL_STATES:
            break
        time.sleep(15)
    if status == "completed":
        print("\nLogs: Studio URL above > Outputs + logs > user_logs/std_log.txt")
        print(f"Next: python azure_ml.py register --from-job {name}")
    else:
        print("\nThe job did not complete; open the Studio URL > Outputs + logs to see why.")


def job_status(args):
    ml = ml_client()
    job = ml.jobs.get(args.name)
    status = status_of(job)
    print(f"{job.name}: {status}")
    print(f"Studio URL: {job.studio_url}")
    if args.wait and status not in TERMINAL_STATES:
        wait_for_job(ml, job.name)
    elif status == "completed":
        print(f"Next: python azure_ml.py register --from-job {job.name}")


def register(args):
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model

    ml = ml_client()
    if args.from_job:
        path = f"azureml://jobs/{args.from_job}/outputs/artifacts/paths/outputs/weather_model.pkl"
    else:
        path = str(MODEL_PATH)
    tags = {"framework": "scikit-learn", "algorithm": "DecisionTreeClassifier"}
    metrics_file = BASE_DIR / "model_metrics.json"
    if metrics_file.exists() and not args.from_job:
        tags["accuracy"] = str(json.loads(metrics_file.read_text())["accuracy"])
    model = ml.models.create_or_update(Model(
        path=path,
        name=AZURE_MODEL_NAME,
        type=AssetTypes.CUSTOM_MODEL,
        description="Decision Tree predicting Rain / No Rain from temperature, humidity, pressure, wind speed",
        tags=tags,
    ))
    print(f"Registered {model.name}:{model.version}")


def latest_version(ml):
    versions = [int(m.version) for m in ml.models.list(name=AZURE_MODEL_NAME)]
    if not versions:
        raise SystemExit(f"No registered versions of {AZURE_MODEL_NAME}")
    return str(max(versions))


def download(args):
    ml = ml_client()
    version = args.version or latest_version(ml)
    with tempfile.TemporaryDirectory() as tmp:
        ml.models.download(name=AZURE_MODEL_NAME, version=version, download_path=tmp)
        pkl = next(Path(tmp).rglob("*.pkl"))
        shutil.copy(pkl, MODEL_PATH)
    print(f"Downloaded {AZURE_MODEL_NAME}:{version} -> {MODEL_PATH}")
    print("Restart the edge module to load it.")


def list_models(args):
    ml = ml_client()
    for m in ml.models.list(name=AZURE_MODEL_NAME):
        print(f"{m.name}:{m.version}  tags={m.tags}")


def main():
    parser = argparse.ArgumentParser(description="Azure ML helper for the weather model")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("submit-job")
    p.add_argument("--wait", action="store_true", help="wait until the job finishes")
    p.set_defaults(func=submit_job)
    p = sub.add_parser("status")
    p.add_argument("name", help="job name printed by submit-job")
    p.add_argument("--wait", action="store_true", help="wait until the job finishes")
    p.set_defaults(func=job_status)
    p = sub.add_parser("register")
    p.add_argument("--from-job", help="register the model output of this Azure ML job")
    p.set_defaults(func=register)
    p = sub.add_parser("download")
    p.add_argument("--version")
    p.set_defaults(func=download)
    sub.add_parser("list").set_defaults(func=list_models)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
