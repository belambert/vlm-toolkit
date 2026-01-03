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
JOB_NAME="imggen-$(date +%Y%m%d-%H%M%S)"
GPU_TYPE="${GPU_TYPE:-}"  # Default: no GPU
GPU_COUNT="${GPU_COUNT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-100}"  # Smaller default disk for CPU-only

# Determine machine type based on GPU type and count
case "${GPU_TYPE}" in
    "")
        # No GPU - use CPU-only machine
        MACHINE_TYPE="n1-standard-4"
        ACCELERATOR_TYPE=""
        ACCELERATOR_COUNT="0"
        ;;
    NVIDIA_TESLA_A100)
        ACCELERATOR_TYPE="NVIDIA_TESLA_A100"
        case "${GPU_COUNT}" in
            1) MACHINE_TYPE="a2-highgpu-1g" ;;
            2) MACHINE_TYPE="a2-highgpu-2g" ;;
            4) MACHINE_TYPE="a2-highgpu-4g" ;;
            8) MACHINE_TYPE="a2-highgpu-8g" ;;
            *) echo "Error: Unsupported GPU count ${GPU_COUNT} for ${GPU_TYPE}. Valid: 1, 2, 4, 8"; exit 1 ;;
        esac
        ACCELERATOR_COUNT="${GPU_COUNT}"
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
        ACCELERATOR_COUNT="${GPU_COUNT}"
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
        ACCELERATOR_COUNT="${GPU_COUNT}"
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
        ACCELERATOR_COUNT="${GPU_COUNT}"
        ;;
    NVIDIA_RTX_A6000)
        ACCELERATOR_TYPE="NVIDIA_RTX_A6000"
        case "${GPU_COUNT}" in
            1|2|4) MACHINE_TYPE="n1-standard-16" ;;
            *) echo "Error: Unsupported GPU count ${GPU_COUNT} for ${GPU_TYPE}. Valid: 1, 2, 4"; exit 1 ;;
        esac
        ACCELERATOR_COUNT="${GPU_COUNT}"
        ;;
    *)
        echo "Error: Unsupported GPU type: ${GPU_TYPE}"
        echo "Valid types: NVIDIA_TESLA_A100, NVIDIA_A100_80GB, NVIDIA_H100_80GB, NVIDIA_L4, NVIDIA_RTX_A6000, or leave empty for CPU-only"
        exit 1
        ;;
esac

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

echo "Submitting Google Cloud Batch job..."
echo "  Job name: ${JOB_NAME}"
echo "  Image: ${IMAGE_URI}"
echo "  Command: $@"
echo "  Machine: ${MACHINE_TYPE}"
if [ -n "${GPU_TYPE}" ]; then
    echo "  GPU: ${GPU_COUNT}x ${GPU_TYPE}"
else
    echo "  GPU: None (CPU-only)"
fi
echo "  Disk: ${BOOT_DISK_SIZE}GB SSD"
if [ -n "${BUCKET}" ]; then
    echo "  GCS Bucket: ${BUCKET} (mounted at /mnt/disks/gcs)"
fi
echo ""

# Create temporary config file
CONFIG_FILE=$(mktemp /tmp/batch-config-XXXXXX.json)
trap "rm -f ${CONFIG_FILE}" EXIT

# Set compute resources based on GPU/CPU mode
if [ -n "${GPU_TYPE}" ]; then
    # GPU jobs need more resources
    CPU_MILLI="16000"
    MEMORY_MIB="65536"
else
    # CPU-only jobs can use smaller resources
    CPU_MILLI="4000"
    MEMORY_MIB="15360"
fi

# Build config JSON for Google Cloud Batch
cat > "${CONFIG_FILE}" << 'OUTER_EOF'
{
  "taskGroups": [
    {
      "taskCount": "1",
      "parallelism": "1",
      "taskSpec": {
        "computeResource": {
          "cpuMilli": "CPU_MILLI_PLACEHOLDER",
          "memoryMib": "MEMORY_MIB_PLACEHOLDER",
          "bootDiskMib": "BOOT_DISK_SIZE_PLACEHOLDER"
        },
        "runnables": [
          {
            "container": {
              "imageUri": "IMAGE_URI_PLACEHOLDER",
              "commands": COMMANDS_PLACEHOLDER,
              "volumes": []
OUTER_EOF

# Build container options
CONTAINER_OPTIONS="--privileged"
if [ -n "${WANDB_API_KEY}" ]; then
    CONTAINER_OPTIONS="${CONTAINER_OPTIONS} --env=WANDB_API_KEY=WANDB_API_KEY_PLACEHOLDER"
fi
if [ -n "${HF_TOKEN}" ]; then
    CONTAINER_OPTIONS="${CONTAINER_OPTIONS} --env=HF_TOKEN=HF_TOKEN_PLACEHOLDER"
fi

cat >> "${CONFIG_FILE}" << 'OUTER_EOF'
              ,
              "options": "CONTAINER_OPTIONS_PLACEHOLDER"
            }
          }
        ],
        "volumes": VOLUMES_PLACEHOLDER
      }
    }
  ],
  "allocationPolicy": {
    "instances": [
      {
OUTER_EOF

# Add GPU configuration if GPU is specified
if [ -n "${GPU_TYPE}" ]; then
    cat >> "${CONFIG_FILE}" << 'OUTER_EOF'
        "installGpuDrivers": true,
        "policy": {
          "machineType": "MACHINE_TYPE_PLACEHOLDER",
          "accelerators": [
            {
              "type": "GPU_TYPE_PLACEHOLDER",
              "count": GPU_COUNT_PLACEHOLDER
            }
          ],
          "bootDisk": {
            "image": "batch-debian"
          }
        }
OUTER_EOF
else
    cat >> "${CONFIG_FILE}" << 'OUTER_EOF'
        "policy": {
          "machineType": "MACHINE_TYPE_PLACEHOLDER",
          "bootDisk": {
            "image": "batch-debian"
          }
        }
OUTER_EOF
fi

cat >> "${CONFIG_FILE}" << 'OUTER_EOF'
      }
    ]
  },
  "logsPolicy": {
    "destination": "CLOUD_LOGGING"
  }
}
OUTER_EOF

# Build commands JSON array
COMMANDS_JSON="["
FIRST=true
for arg in "${ALL_ARGS[@]}"; do
    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        COMMANDS_JSON+=","
    fi
    # Escape quotes in the argument
    ESCAPED_ARG=$(echo "$arg" | sed 's/"/\\"/g')
    COMMANDS_JSON+="\"${ESCAPED_ARG}\""
done
COMMANDS_JSON+="]"

# Build volumes JSON
if [ -n "${BUCKET}" ]; then
    VOLUMES_JSON="[{\"gcs\":{\"remotePath\":\"${BUCKET}\"},\"mountPath\":\"/mnt/disks/gcs\"}]"
else
    VOLUMES_JSON="[]"
fi

# Convert GPU_TYPE to Batch format
if [ -n "${GPU_TYPE}" ]; then
    case "${GPU_TYPE}" in
        NVIDIA_TESLA_A100) BATCH_GPU_TYPE="nvidia-tesla-a100" ;;
        NVIDIA_A100_80GB) BATCH_GPU_TYPE="nvidia-a100-80gb" ;;
        NVIDIA_H100_80GB) BATCH_GPU_TYPE="nvidia-h100-80gb" ;;
        NVIDIA_L4) BATCH_GPU_TYPE="nvidia-l4" ;;
        NVIDIA_RTX_A6000) BATCH_GPU_TYPE="nvidia-tesla-a100" ;; # Fallback
        *) BATCH_GPU_TYPE="nvidia-l4" ;;
    esac
else
    BATCH_GPU_TYPE=""
fi

# Replace placeholders
sed -i.bak \
    -e "s|IMAGE_URI_PLACEHOLDER|${IMAGE_URI}|g" \
    -e "s|COMMANDS_PLACEHOLDER|${COMMANDS_JSON}|g" \
    -e "s|VOLUMES_PLACEHOLDER|${VOLUMES_JSON}|g" \
    -e "s|MACHINE_TYPE_PLACEHOLDER|${MACHINE_TYPE}|g" \
    -e "s|GPU_TYPE_PLACEHOLDER|${BATCH_GPU_TYPE}|g" \
    -e "s|GPU_COUNT_PLACEHOLDER|${GPU_COUNT}|g" \
    -e "s|CPU_MILLI_PLACEHOLDER|${CPU_MILLI}|g" \
    -e "s|MEMORY_MIB_PLACEHOLDER|${MEMORY_MIB}|g" \
    -e "s|BOOT_DISK_SIZE_PLACEHOLDER|$((BOOT_DISK_SIZE * 1024))|g" \
    -e "s|CONTAINER_OPTIONS_PLACEHOLDER|${CONTAINER_OPTIONS}|g" \
    -e "s|WANDB_API_KEY_PLACEHOLDER|${WANDB_API_KEY}|g" \
    -e "s|HF_TOKEN_PLACEHOLDER|${HF_TOKEN}|g" \
    "${CONFIG_FILE}"
rm -f "${CONFIG_FILE}.bak"

# Execute the command
gcloud batch jobs submit \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --job-prefix="${IMAGE_NAME}" \
    --config="${CONFIG_FILE}"

echo ""
echo "Job submitted successfully!"
echo "Monitor at: https://console.cloud.google.com/batch/jobs?project=${PROJECT_ID}"
