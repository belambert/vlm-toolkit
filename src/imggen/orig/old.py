import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

MODEL_ID = "stabilityai/stable-diffusion-2-1"


def main():
    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    print("setting up pipline...")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe = pipe.to("cuda")

    print("generating...")
    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]

    print("saving...")
    image.save("astronaut_rides_horse.png")


if __name__ == "__main__":
    main()
