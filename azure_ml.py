"""Azure Machine Learning helper: train in the cloud, register, deploy and fetch the model.

Requires `pip install -r requirements-azure.txt`, `az login`, and the AZURE_* values in .env.

Usage:
  python azure_ml.py submit-job [--wait]          # run train_model.py on the Azure ML compute
  python azure_ml.py status <job> [--wait]        # check (or wait for) a submitted job
  python azure_ml.py register --from-job <job>    # register the model produced by a cloud job
  python azure_ml.py register                     # or register the local energy_model.pkl
  python azure_ml.py list                         # list registered versions
  python azure_ml.py download [--version N]       # pull a registered model for the edge module

  python azure_ml.py deploy-endpoint              # deploy the latest model as a REST endpoint
  python azure_ml.py endpoint-info [--save]       # print (and save to .env) its URL and key
  python azure_ml.py test-endpoint                # send azure_deploy/sample_request.json
  python azure_ml.py endpoint-logs                # container logs, if the deployment fails
  python azure_ml.py delete-endpoint              # delete it again (stops the charges)
"""
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

from config import (
    AZURE_ENDPOINT_INSTANCE_TYPE, AZURE_ENDPOINT_NAME, AZURE_ML_COMPUTE, AZURE_ML_WORKSPACE,
    AZURE_MODEL_NAME, AZURE_RESOURCE_GROUP, AZURE_SUBSCRIPTION_ID, BASE_DIR, MODEL_PATH,
)

JOB_FILES = ["train_model.py", "energy_data.csv"]
ENDPOINT_FILES = ["azure_deploy/score.py", "features.py"]
TRAINING_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
# Same base image as training (already proven to build); conda adds azureml-inference-server-http
INFERENCE_IMAGE = TRAINING_IMAGE
DEPLOYMENT_NAME = "blue"
TERMINAL_STATES = {"completed", "failed", "canceled", "notresponding"}


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


def status_of(job):
    # Newer SDKs return an enum (JobStatus.COMPLETED), older ones a string ("Completed")
    status = getattr(job.status, "value", job.status)
    return str(status).split(".")[-1].lower()


# ---------------------------------------------------------------- training job

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
                name="energy-ml-environment",
                conda_file=str(BASE_DIR / "conda.yml"),
                image=TRAINING_IMAGE,
            ),
            compute=AZURE_ML_COMPUTE,
            experiment_name="energy-edge-ai",
            display_name="energy-consumption-model-training",
        )
        job = ml.jobs.create_or_update(job)
    print(f"Submitted job: {job.name}")
    print(f"Studio URL:    {job.studio_url}")
    if args.wait or args.register:
        if wait_for_job(ml, job.name) == "completed" and args.register:
            args.from_job = job.name
            register(args)


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
    return status


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


# ---------------------------------------------------------------- model registry

def register(args):
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model

    ml = ml_client()
    if args.from_job:
        path = f"azureml://jobs/{args.from_job}/outputs/artifacts/paths/outputs/energy_model.pkl"
    else:
        path = str(MODEL_PATH)
    tags = {"framework": "scikit-learn", "algorithm": "RandomForestRegressor", "target": "kWh next hour"}
    metrics_file = BASE_DIR / "model_metrics.json"
    if metrics_file.exists() and not args.from_job:
        m = json.loads(metrics_file.read_text())
        tags.update(mae_kwh=str(m["mae"]), r2=str(m["r2"]))
    model = ml.models.create_or_update(Model(
        path=path,
        name=AZURE_MODEL_NAME,
        type=AssetTypes.CUSTOM_MODEL,
        description="Random Forest predicting a home's electricity use (kWh) for the coming hour "
                    "from time of day, temperature, humidity, occupancy and last hour's use",
        tags=tags,
    ))
    print(f"Registered {model.name}:{model.version}")


def latest_version(ml):
    versions = [int(m.version) for m in ml.models.list(name=AZURE_MODEL_NAME)]
    if not versions:
        raise SystemExit(f"No registered versions of {AZURE_MODEL_NAME}; run register first")
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


# ---------------------------------------------------------------- online endpoint

def deploy_endpoint(args):
    from azure.ai.ml.entities import (
        CodeConfiguration, Environment, ManagedOnlineDeployment, ManagedOnlineEndpoint,
    )

    ml = ml_client()
    version = args.version or latest_version(ml)
    model = ml.models.get(AZURE_MODEL_NAME, version)

    endpoint = ManagedOnlineEndpoint(
        name=AZURE_ENDPOINT_NAME,
        auth_mode="key",
        description="Next-hour household electricity consumption (kWh)",
    )
    print(f"[1/3] Creating endpoint {AZURE_ENDPOINT_NAME} (about 2 min) ...")
    ml.online_endpoints.begin_create_or_update(endpoint).result()

    # A deployment that failed during provisioning cannot be updated, only deleted and re-created
    from azure.core.exceptions import ResourceNotFoundError
    try:
        existing = ml.online_deployments.get(name=DEPLOYMENT_NAME, endpoint_name=AZURE_ENDPOINT_NAME)
        state = getattr(existing.provisioning_state, "value", existing.provisioning_state)  # enum or str
        if str(state).lower() == "failed":
            print(f"      Deleting failed deployment '{DEPLOYMENT_NAME}' from the previous attempt (a few min) ...")
            ml.online_deployments.begin_delete(name=DEPLOYMENT_NAME, endpoint_name=AZURE_ENDPOINT_NAME).result()
    except ResourceNotFoundError:
        pass

    print(f"[2/3] Deploying {AZURE_MODEL_NAME}:{version} as '{DEPLOYMENT_NAME}' on 1 x {args.instance_type} "
          "(10-20 min the first time) ...")
    with tempfile.TemporaryDirectory() as src:  # upload only the scoring code
        for name in ENDPOINT_FILES:
            shutil.copy(BASE_DIR / name, src)
        deployment = ManagedOnlineDeployment(
            name=DEPLOYMENT_NAME,
            endpoint_name=AZURE_ENDPOINT_NAME,
            model=model,
            environment=Environment(conda_file=str(BASE_DIR / "azure_deploy" / "conda.yml"), image=INFERENCE_IMAGE),
            code_configuration=CodeConfiguration(code=src, scoring_script="score.py"),
            instance_type=args.instance_type,
            instance_count=1,
        )
        ml.online_deployments.begin_create_or_update(deployment).result()

    print("[3/3] Routing 100% of traffic to the deployment ...")
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml.online_endpoints.begin_create_or_update(endpoint).result()
    print("\nDeployed. Next:")
    print("  python azure_ml.py test-endpoint")
    print("  python azure_ml.py endpoint-info --save")
    print("  python edge_module.py --predictor endpoint")
    print("Remember: the endpoint is billed every hour it exists -> python azure_ml.py delete-endpoint")


def set_env_value(key, value):
    env = BASE_DIR / ".env"
    lines = env.read_text().splitlines() if env.exists() else []
    lines = [line for line in lines if not line.startswith(f"{key}=")] + [f"{key}={value}"]
    env.write_text("\n".join(lines) + "\n")


def endpoint_info(args):
    ml = ml_client()
    endpoint = ml.online_endpoints.get(AZURE_ENDPOINT_NAME)
    key = ml.online_endpoints.get_keys(AZURE_ENDPOINT_NAME).primary_key
    print(f"Endpoint:    {endpoint.name}  ({endpoint.provisioning_state}, traffic {endpoint.traffic})")
    print(f"Scoring URL: {endpoint.scoring_uri}")
    print(f"Key:         {key[:6]}...  (full key {'saved to .env' if args.save else 'shown with --save'})")
    if args.save:
        set_env_value("AZURE_ENDPOINT_URL", endpoint.scoring_uri)
        set_env_value("AZURE_ENDPOINT_KEY", key)
        print("Saved AZURE_ENDPOINT_URL and AZURE_ENDPOINT_KEY to .env")


def test_endpoint(args):
    ml = ml_client()
    request_file = BASE_DIR / "azure_deploy" / "sample_request.json"
    print("Request: ", json.dumps(json.loads(request_file.read_text())))
    response = ml.online_endpoints.invoke(endpoint_name=AZURE_ENDPOINT_NAME, request_file=str(request_file))
    print("Response:", response)


def endpoint_logs(args):
    ml = ml_client()
    print(ml.online_deployments.get_logs(name=DEPLOYMENT_NAME, endpoint_name=AZURE_ENDPOINT_NAME, lines=args.lines))


def delete_endpoint(args):
    ml = ml_client()
    print(f"Deleting endpoint {AZURE_ENDPOINT_NAME} (a few minutes) ...")
    ml.online_endpoints.begin_delete(name=AZURE_ENDPOINT_NAME).result()
    print("Deleted.")


def main():
    parser = argparse.ArgumentParser(description="Azure ML helper for the energy model")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("submit-job")
    p.add_argument("--wait", action="store_true", help="wait until the job finishes")
    p.add_argument("--register", action="store_true", help="wait, then register the trained model")
    p.set_defaults(func=submit_job)
    p = sub.add_parser("status")
    p.add_argument("name", help="job name printed by submit-job")
    p.add_argument("--wait", action="store_true", help="wait until the job finishes")
    p.set_defaults(func=job_status)
    p = sub.add_parser("register")
    p.add_argument("--from-job", help="register the model output of this Azure ML job")
    p.set_defaults(func=register)
    sub.add_parser("list").set_defaults(func=list_models)
    p = sub.add_parser("download")
    p.add_argument("--version")
    p.set_defaults(func=download)
    p = sub.add_parser("deploy-endpoint")
    p.add_argument("--version", help="model version (default: latest)")
    p.add_argument("--instance-type", default=AZURE_ENDPOINT_INSTANCE_TYPE)
    p.set_defaults(func=deploy_endpoint)
    p = sub.add_parser("endpoint-info")
    p.add_argument("--save", action="store_true", help="write the URL and key to .env")
    p.set_defaults(func=endpoint_info)
    sub.add_parser("test-endpoint").set_defaults(func=test_endpoint)
    p = sub.add_parser("endpoint-logs")
    p.add_argument("--lines", type=int, default=100)
    p.set_defaults(func=endpoint_logs)
    sub.add_parser("delete-endpoint").set_defaults(func=delete_endpoint)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
