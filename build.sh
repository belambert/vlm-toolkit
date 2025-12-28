#!/bin/bash
set -e

# Configuration - update these for your project
PROJECT_ID="${GCP_PROJECT_ID:-llm-exp-405305}"
# Use multi-regional location for availability across regions
# Options: us, europe, asia, or specific regions like us-central1
REGION="${GCP_REGION:-us}"
IMAGE_NAME="finetune"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Full image URI for Artifact Registry
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/finetune/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building Docker image..."
docker build --platform linux/amd64 -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo "Tagging image for Artifact Registry..."
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${IMAGE_URI}"

echo "Pushing image to Artifact Registry..."
docker push "${IMAGE_URI}"

echo ""
echo "Image successfully pushed to:"
echo "  ${IMAGE_URI}"
echo ""
echo "Note: Make sure you have:"
echo "  1. Created a multi-regional Artifact Registry repository:"
echo "     gcloud artifacts repositories create finetune --repository-format=docker --location=${REGION}"
echo "  2. Configured Docker authentication:"
echo "     gcloud auth configure-docker ${REGION}-docker.pkg.dev"
echo ""
echo "Multi-regional locations (us, europe, asia) make images available across all regions in that area."
