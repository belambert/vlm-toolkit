#!/bin/bash

# Required Configuration
export PROJECT_ID="llm-exp-405305"

REPO_NAME="imggen"
IMAGE_NAME="imggen"
IMAGE_TAG="${IMAGE_TAG:-latest}"

export IMAGE_URI="us-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# Optional Configuration
export REGION="us-central1"
# export GPU_TYPE="nvidia-l4"      # or nvidia-tesla-a100, nvidia-tesla-t4, etc.
# export GPU_TYPE="nvidia-tesla-t4"
# export GPU_TYPE="nvidia-tesla-a100"
export GPU_TYPE="nvidia-a100-80gb"
# export BOOT_DISK_SIZE=200
export GPU_COUNT=2

BUCKET=imggen

# Source the library from the submodule
source "gcp-batch/lib/gcp_batch.sh"

# Submit your command
submit_batch_job $@
