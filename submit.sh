#!/bin/bash
set -e

# Configuration - update these for your project
PROJECT_ID="${GCP_PROJECT_ID:-llm-exp-405305}"
REGION="${GCP_REGION:-us-central1}"  # multi-regional for availability across regions
IMAGE_NAME="imggen"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Full image URI for Artifact Registry
IMAGE_URI="us-docker.pkg.dev/${PROJECT_ID}/imggen/${IMAGE_NAME}:${IMAGE_TAG}"

# Job configuration
JOB_NAME="finetune-$(date +%Y%m%d-%H%M%S)"
GPU_TYPE="${GPU_TYPE:-NVIDIA_A100_80GB}"
GPU_COUNT="${GPU_COUNT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-1000}"

# Determine machine type based on GPU type and count
case "${GPU_TYPE}" in
    NVIDIA_TESLA_A100)
        ACCELERATOR_TYPE="NVIDIA_TESLA_A100"
        case "${GPU_COUNT}" in
            1) MACHINE_TYPE="a2-highgpu-1g" ;;
            2) MACHINE_TYPE="a2-highgpu-2g" ;;
            4) MACHINE_TYPE="a2-highgpu-4g" ;;
            8) MACHINE_TYPE="a2-highgpu-8g" ;;
            *) echo "Error: Unsupported GPU count ${GPU_COUNT} for ${GPU_TYPE}. Valid: 1, 2, 4, 8"; exit 1 ;;
        esac
        ;;
    NVIDIA_A100_80GB)
        ACCELERATOR_TYPE="NVIDIA_A100_80GB"
        case "${GPU_COUNT}" in
            1) MACHINE_TYPE="a2-ultragpu-1g" ;;
            2) MACHINE_TYPE="a2-ultragpu-2g" ;;
            4) MACHINE_TYPE="a2-ultragpu-4g" ;;
            8) MACHINE_TYPE="a2-ultragpu-8g" ;;
            *) echo "Error: Unsupported GPU count ${GPU_COUNT} for ${GPU_TYPE}. Valid: 1, 2, 4, 8"; exit 1 ;;
        esac
        ;;
    NVIDIA_H100_80GB)
        ACCELERATOR_TYPE="NVIDIA_H100_80GB"
        case "${GPU_COUNT}" in
            1) MACHINE_TYPE="a3-highgpu-1g" ;;
            2) MACHINE_TYPE="a3-highgpu-2g" ;;
            4) MACHINE_TYPE="a3-highgpu-4g" ;;
            8) MACHINE_TYPE="a3-highgpu-8g" ;;
            *) echo "Error: Unsupported GPU count ${GPU_COUNT} for ${GPU_TYPE}. Valid: 1, 2, 4, 8"; exit 1 ;;
        esac
        ;;
    NVIDIA_L4)
        ACCELERATOR_TYPE="NVIDIA_L4"
        case "${GPU_COUNT}" in
            1) MACHINE_TYPE="g2-standard-12" ;;
            2) MACHINE_TYPE="g2-standard-24" ;;
            4) MACHINE_TYPE="g2-standard-48" ;;
            8) MACHINE_TYPE="g2-standard-96" ;;
            *) echo "Error: Unsupported GPU count ${GPU_COUNT} for ${GPU_TYPE}. Valid: 1, 2, 4, 8"; exit 1 ;;
        esac
        ;;
    NVIDIA_RTX_A6000)
        ACCELERATOR_TYPE="NVIDIA_RTX_A6000"
        case "${GPU_COUNT}" in
            1|2|4) MACHINE_TYPE="n1-standard-16" ;;
            *) echo "Error: Unsupported GPU count ${GPU_COUNT} for ${GPU_TYPE}. Valid: 1, 2, 4"; exit 1 ;;
        esac
        ;;
    *)
        echo "Error: Unsupported GPU type: ${GPU_TYPE}"
        echo "Valid types: NVIDIA_TESLA_A100, NVIDIA_A100_80GB, NVIDIA_H100_80GB, NVIDIA_L4, NVIDIA_RTX_A6000"
        exit 1
        ;;
esac

ACCELERATOR_COUNT="${GPU_COUNT}"

# Check for environment variables
if [ -z "${WANDB_API_KEY}" ]; then
    echo "Warning: WANDB_API_KEY not set. Set it with: export WANDB_API_KEY=your-key"
fi
if [ -z "${HF_TOKEN}" ]; then
    echo "Warning: HF_TOKEN not set. Set it with: export HF_TOKEN=your-token"
fi

# Get all command arguments
# Usage: ./submit.sh ./bin/check-inference.sh
# Usage: ./submit.sh uv run grpo --lr 1e-4
if [ $# -lt 1 ]; then
    echo "Usage: $0 <command...>"
    echo "Example: $0 ./bin/check-inference.sh"
    echo "Example: $0 uv run grpo --lr 1e-4"
    exit 1
fi

ALL_ARGS=("$@")

echo "Submitting Vertex AI custom training job..."
echo "  Job name: ${JOB_NAME}"
echo "  Image: ${IMAGE_URI}"
echo "  Command: $@"
echo "  Machine: ${MACHINE_TYPE}"
echo "  GPU: ${GPU_COUNT}x ${GPU_TYPE}"
echo "  Disk: ${BOOT_DISK_SIZE}GB SSD"
echo ""

# Create temporary config file
CONFIG_FILE=$(mktemp /tmp/vertex-config-XXXXXX.yaml)
trap "rm -f ${CONFIG_FILE}" EXIT

# Build config YAML
cat > "${CONFIG_FILE}" << EOF
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE_TYPE}
      acceleratorType: ${ACCELERATOR_TYPE}
      acceleratorCount: ${ACCELERATOR_COUNT}
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: ${BOOT_DISK_SIZE}
    replicaCount: 1
    containerSpec:
      imageUri: ${IMAGE_URI}
EOF

# Add environment variables if set
if [ -n "${WANDB_API_KEY}" ] || [ -n "${HF_TOKEN}" ]; then
    cat >> "${CONFIG_FILE}" << EOF
      env:
EOF
    if [ -n "${WANDB_API_KEY}" ]; then
        cat >> "${CONFIG_FILE}" << EOF
        - name: WANDB_API_KEY
          value: "${WANDB_API_KEY}"
EOF
    fi
    if [ -n "${HF_TOKEN}" ]; then
        cat >> "${CONFIG_FILE}" << EOF
        - name: HF_TOKEN
          value: "${HF_TOKEN}"
EOF
    fi
fi

# Add command
cat >> "${CONFIG_FILE}" << EOF
      command:
EOF
for arg in "${ALL_ARGS[@]}"; do
    echo "        - \"${arg}\"" >> "${CONFIG_FILE}"
done

# Execute the command
gcloud ai custom-jobs create \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --display-name="${JOB_NAME}" \
    --config="${CONFIG_FILE}"

echo ""
echo "Job submitted successfully!"
echo "Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
