# "poetry run python ./imggen/train_lora.py /mnt/disks/imggen/lora_data/data_set_sm/ --num-train-epochs 1 --train-batch-size 1 --output-dir /mnt/disks/imggen/loras/"
# "poetry run train-lora /mnt/disks/imggen/lora_data/bluejay/ --num-train-epochs 200 --train-batch-size 1 --output-dir /mnt/disks/imggen/loras/bluejay --max-training-data 1 --learning-rate 1e-5 --rank 128"

TAG=latest
# TAG="556bed6f13a88b5ec8612aee671d12a4e65624eb"  # before big refactor
# TAG="4931d95371b3637a73ea51aac895f97247325f93" # first time doing generation

gcloud batch jobs submit --job-prefix imggen-train-lora --location us-central1 --config - <<EOD
{
  "taskGroups": [
    {
      "taskCount": "1",
      "parallelism": "1",
      "taskSpec": {
        "computeResource": {
          "cpuMilli": "4000",
          "memoryMib": "32000",
          "bootDiskMib": "50000"
        },
        "runnables": [
          {
            "container": {
              "imageUri": "us-docker.pkg.dev/llm-exp-405305/imggen/imggen:$TAG",
              "entrypoint": "/bin/sh",
              "commands": [
                "-c",
                "./run-sdxl.sh"
              ],
              "volumes": []
            }
          }
        ],
        "volumes": [
          {
            "gcs": {
              "remotePath": "imggen"
            },
            "mountPath": "/mnt/disks/imggen"
          }
        ]
      }
    }
  ],
  "allocationPolicy": {
    "instances": [
      {
        "installGpuDrivers": true,
        "policy": {
          "accelerators": [
            {
              "type": "nvidia-l4",
              "count": 1
            }
          ]
        }
      }
    ]
  },
  "logsPolicy": {
    "destination": "CLOUD_LOGGING"
  }
}
