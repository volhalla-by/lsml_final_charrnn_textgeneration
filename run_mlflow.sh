#!/usr/bin/env bash

mlflow server \
--backend-store-uri="sqlite:////app/mlflow/mlflow.db" \
--default-artifact-root="/app/mlflow/artifacts" \
--host 0.0.0.0 \
--port 5003 \
--workers 2
