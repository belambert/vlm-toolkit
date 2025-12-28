# "poetry run python ./imggen/train_lora.py /mnt/disks/imggen/lora_data/data_set_sm/ --num-train-epochs 1 --train-batch-size 1 --output-dir /mnt/disks/imggen/loras/"
# "poetry run train-lora /mnt/disks/imggen/lora_data/bluejay/ --num-train-epochs 200 --train-batch-size 1 --output-dir /mnt/disks/imggen/loras/bluejay --max-training-data 1 --learning-rate 1e-5 --rank 128"

TAG=latest

# MODEL="runwayml/stable-diffusion-v1-5"
# MODEL="stabilityai/stable-diffusion-xl-base-1.0"
MODEL="stablediffusionapi/sdxxxl"

GPU="nvidia-tesla-a100"
# GPU="nvidia-l4"
RANK=128
EPOCHS=10
BATCH_SIZE=2


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
                "poetry run train-lora /mnt/disks/imggen/lora_data/rcg --num-train-epochs $EPOCHS --train-batch-size $BATCH_SIZE --output-dir /mnt/disks/imggen/loras/rcg4 --learning-rate 1e-6 --rank $RANK --model $MODEL --resolution 1024"
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
              "type": "$GPU",
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
