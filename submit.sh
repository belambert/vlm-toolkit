#!/bin/bash
set -e

PROJECT_ID="${GCP_PROJECT_ID:-llm-exp-405305}"
REGION="${GCP_REGION:-us-central1}"
IMAGE_NAME="imggen"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Full image URI for Artifact Registry
IMAGE_URI="us-docker.pkg.dev/${PROJECT_ID}/imggen/${IMAGE_NAME}:${IMAGE_TAG}"

# Job configuration
JOB_NAME="imggen-$(date +%Y%m%d-%H%M%S)"
GPU_TYPE="${GPU_TYPE:-}"  # Default: no GPU
GPU_COUNT="${GPU_COUNT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-100}"

# Determine machine type based on GPU type and count
case "${GPU_TYPE}" in
    "")
        # No GPU - use CPU-only machine
        MACHINE_TYPE="n1-standard-4"
        ACCELERATOR_COUNT="0"
        ;;
    nvidia-tesla-a100)
        case "${GPU_COUNT}" in
            1) MACHINE_TYPE="a2-highgpu-1g" ;;
            2) MACHINE_TYPE="a2-highgpu-2g" ;;
            4) MACHINE_TYPE="a2-highgpu-4g" ;;
            8) MACHINE_TYPE="a2-highgpu-8g" ;;
            *) echo "Error: Unsupported GPU count ${GPU_COUNT} for ${GPU_TYPE}. Valid: 1, 2, 4, 8"; exit 1 ;;
        esac
        ACCELERATOR_COUNT="${GPU_COUNT}"
        ;;
    nvidia-a100-80gb)
        case "${GPU_COUNT}" in
            1) MACHINE_TYPE="a2-ultragpu-1g" ;;
            2) MACHINE_TYPE="a2-ultragpu-2g" ;;
            4) MACHINE_TYPE="a2-ultragpu-4g" ;;
            8) MACHINE_TYPE="a2-ultragpu-8g" ;;
            *) echo "Error: Unsupported GPU count ${GPU_COUNT} for ${GPU_TYPE}. Valid: 1, 2, 4, 8"; exit 1 ;;
        esac
        ACCELERATOR_COUNT="${GPU_COUNT}"
        ;;
    nvidia-h100-80gb)
        case "${GPU_COUNT}" in
            1) MACHINE_TYPE="a3-highgpu-1g" ;;
            2) MACHINE_TYPE="a3-highgpu-2g" ;;
            4) MACHINE_TYPE="a3-highgpu-4g" ;;
            8) MACHINE_TYPE="a3-highgpu-8g" ;;
            *) echo "Error: Unsupported GPU count ${GPU_COUNT} for ${GPU_TYPE}. Valid: 1, 2, 4, 8"; exit 1 ;;
        esac
        ACCELERATOR_COUNT="${GPU_COUNT}"
        ;;
    nvidia-l4)
        case "${GPU_COUNT}" in
            1) MACHINE_TYPE="g2-standard-12" ;;
            2) MACHINE_TYPE="g2-standard-24" ;;
            4) MACHINE_TYPE="g2-standard-48" ;;
            8) MACHINE_TYPE="g2-standard-96" ;;
            *) echo "Error: Unsupported GPU count ${GPU_COUNT} for ${GPU_TYPE}. Valid: 1, 2, 4, 8"; exit 1 ;;
        esac
        ACCELERATOR_COUNT="${GPU_COUNT}"
        ;;
    nvidia-tesla-t4)
        case "${GPU_COUNT}" in
            1|2|4) MACHINE_TYPE="n1-standard-16" ;;
            *) echo "Error: Unsupported GPU count ${GPU_COUNT} for ${GPU_TYPE}. Valid: 1, 2, 4"; exit 1 ;;
        esac
        ACCELERATOR_COUNT="${GPU_COUNT}"
        ;;
    *)
        echo "Error: Unsupported GPU type: ${GPU_TYPE}"
        echo "Valid types: nvidia-tesla-a100, nvidia-a100-80gb, nvidia-h100-80gb, nvidia-l4, nvidia-tesla-t4, or leave empty for CPU-only"
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
echo "  GPU: ${GPU_COUNT}x ${GPU_TYPE}"
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
    HAS_GPU="true"
else
    # CPU-only jobs can use smaller resources
    CPU_MILLI="4000"
    MEMORY_MIB="15360"
    HAS_GPU=""
fi

# Build container options
CONTAINER_OPTIONS="--privileged"
if [ -n "${WANDB_API_KEY}" ]; then
    CONTAINER_OPTIONS="${CONTAINER_OPTIONS} --env=WANDB_API_KEY=${WANDB_API_KEY}"
fi
if [ -n "${HF_TOKEN}" ]; then
    CONTAINER_OPTIONS="${CONTAINER_OPTIONS} --env=HF_TOKEN=${HF_TOKEN}"
fi

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

# Calculate boot disk size in MiB
BOOT_DISK_MIB=$((BOOT_DISK_SIZE * 1024))

# Create JSON data file for mustache
MUSTACHE_DATA=$(mktemp /tmp/mustache-data-XXXXXX.json)
trap "rm -f ${CONFIG_FILE} ${MUSTACHE_DATA}" EXIT

# Escape JSON for embedding in JSON string
COMMANDS_JSON_ESCAPED=$(echo "${COMMANDS_JSON}" | sed 's/"/\\"/g')
VOLUMES_JSON_ESCAPED=$(echo "${VOLUMES_JSON}" | sed 's/"/\\"/g')

cat > "${MUSTACHE_DATA}" << EOF
{
  "IMAGE_URI": "${IMAGE_URI}",
  "COMMANDS_JSON": "${COMMANDS_JSON_ESCAPED}",
  "VOLUMES_JSON": "${VOLUMES_JSON_ESCAPED}",
  "MACHINE_TYPE": "${MACHINE_TYPE}",
  "GPU_TYPE": "${GPU_TYPE}",
  "GPU_COUNT": ${GPU_COUNT},
  "CPU_MILLI": "${CPU_MILLI}",
  "MEMORY_MIB": "${MEMORY_MIB}",
  "BOOT_DISK_MIB": "${BOOT_DISK_MIB}",
  "CONTAINER_OPTIONS": "${CONTAINER_OPTIONS}",
  "HAS_GPU": $([ -n "${GPU_TYPE}" ] && echo "true" || echo "false")
}
EOF

# Render template with mustache
mustache "${MUSTACHE_DATA}" templates/batch-job.json > "${CONFIG_FILE}"

# Execute the command
gcloud batch jobs submit \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --job-prefix="${IMAGE_NAME}" \
    --config="${CONFIG_FILE}"

echo ""
echo "Job submitted successfully!"
echo "Monitor at: https://console.cloud.google.com/batch/jobs?project=${PROJECT_ID}"
