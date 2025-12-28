gcloud batch jobs submit --job-prefix imggen --location us-central1 --config - <<EOD
{
  "taskGroups": [
    {
      "taskCount": "1",
      "parallelism": "1",
      "taskSpec": {
        "computeResource": {
          "cpuMilli": "4000",
          "memoryMib": "32000",
          "bootDiskMib": "20000"
        },
        "runnables": [
          {
            "container": {
              "imageUri": "us-docker.pkg.dev/llm-exp-405305/imggen/imggen:latest",
              "entrypoint": "/bin/sh",
              "commands": [
                "-c",
                "poetry run python ./imggen/main.py stablediffusionapi/sdxxxl /mnt/disks/imggen/prompts/prompts4.txt --img-per-prompt 3 --output-folder /mnt/disks/imggen/images/raw_sdxxxl"
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
