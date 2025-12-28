
#TAG=latest
TAG=915342bbad205b26f6c9248188db416781de2eb9

CMD="poetry run python ./imggen/sd3/sd3.py --output-folder /mnt/disks/imggen/images/sd3 --prompt \\\"hot girls on the beach\\\""

gcloud batch jobs submit --job-prefix imggen-cmd --location us-central1 --config - <<EOD
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
                "$CMD"
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
