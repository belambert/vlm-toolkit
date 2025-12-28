import math
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import typer
from datasets import Dataset, load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision import transforms as tv_transforms
from transformers import CLIPTextModel, CLIPTokenizer

from imggen.util import get_device

# load an image dataset: https://huggingface.co/docs/datasets/image_dataset
# training txt2image: https://huggingface.co/docs/diffusers/training/text2image
# training lora: https://huggingface.co/docs/diffusers/training/lora
# pylint: disable-next=line-too-long
# example training script: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
# what I pasted below:
# pylint: disable-next=line-too-long
# example lora training script: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

# logger = get_logger(__name__, log_level="INFO")

# MODEL = "runwayml/stable-diffusion-v1-5"
# or maybe:
# stabilityai/stable-diffusion-xl-base-1.0


# pylint: disable-next=too-many-arguments,too-many-branches,too-many-statements
def main(
    data_dir: Path,
    model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    rank: int = 8,
    learning_rate: float = 1e-4,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    caption_column: str = "caption",
    image_column: str = "image",
    resolution: int = 1024,
    grad_acc_steps: int = 1,
    output_dir: Path = Path("output"),
    center_crop: bool = False,
    random_flip: bool = False,
    # seed = None,
    train_batch_size: int = 16,
    dataloader_num_workers: int = 0,
    max_train_steps: Optional[int] = None,
    num_train_epochs: int = 100,
    # 'The scheduler type to use.
    # Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
    # ' "constant", "constant_with_warmup"]'
    lr_scheduler_name: str = "constant",
    lr_warmup_steps: int = 500,
    noise_offset: int = 0,
    max_training_data: Optional[int] = None,
    # prediction_type=None,
    # snr_gamma=None,
    # wandb: bool = True,
):

    trainer = LoraTrainer(
        data_dir=data_dir,
        model=model,
        rank=rank,
        learning_rate=learning_rate,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_weight_decay=adam_weight_decay,
        adam_epsilon=adam_epsilon,
        caption_column=caption_column,
        image_column=image_column,
        resolution=resolution,
        grad_acc_steps=grad_acc_steps,
        output_dir=output_dir,
        center_crop=center_crop,
        random_flip=random_flip,
        # seed = None,
        train_batch_size=train_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        max_train_steps=max_train_steps,
        num_train_epochs=num_train_epochs,
        # 'The scheduler type to use.
        # Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        # ' "constant", "constant_with_warmup"]'
        lr_scheduler_name=lr_scheduler_name,
        lr_warmup_steps=lr_warmup_steps,
        noise_offset=noise_offset,
        # it doesn't like these for some reason...
        # prediction_type=prediction_type,
        # snr_gamma=snr_gamma,
        max_training_data=max_training_data,
        # wandb=wb.init(project="lora_train", mode="online" if wandb else "disabled"),
    )

    trainer.do_training()


@dataclass
# pylint: disable-next=too-many-instance-attributes
class LoraTrainer:

    data_dir: Path
    model: str
    rank: int = 8
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    caption_column: str = "caption"
    image_column: str = "image"
    resolution: int = 1024
    grad_acc_steps: int = 1
    output_dir: Path = Path("output")
    center_crop: bool = False
    random_flip: bool = False
    # seed = None,
    train_batch_size: int = 16
    dataloader_num_workers: int = 0
    max_train_steps: Optional[int] = None
    num_train_epochs: int = 100
    # 'The scheduler type to use.
    # Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
    # ' "constant", "constant_with_warmup"]'
    lr_scheduler_name: str = "constant"
    lr_warmup_steps: int = 500
    noise_offset: int = 0
    prediction_type = None
    snr_gamma = None
    max_training_data: Optional[int] = None
    wandb = None

    # these don't need inits...
    noise_scheduler: DDPMScheduler = field(init=False)
    tokenizer: CLIPTokenizer = field(init=False)
    text_encoder: CLIPTextModel = field(init=False)
    vae: AutoencoderKL = field(init=False)
    unet: UNet2DConditionModel = field(init=False)
    device: torch.device = field(init=False)
    dataset: Dataset = field(init=False)
    train_dataset: Dataset = field(init=False)
    dataloader: torch.utils.data.DataLoader = field(init=False)
    optimizer: Optimizer = field(init=False)
    lr_scheduler: LRScheduler = field(init=False)
    pipe: StableDiffusionXLPipeline = field(init=False)

    # do we need something like this...
    # , torch_dtype=torch.float16, use_safetensors=True, variant="fp16"

    def __post_init__(self):
        # getting RuntimeError: Input type (c10::Half) and bias type (float) should be the same
        self.weight_dtype = torch.float32
        self.device = get_device()
        print(self.device)

    def do_training(self):
        self.load_models()
        self.load_data()
        self.training_prep()
        self.training_loop()
        self.save_model()
        self.validation()

    def load_models(self):
        print("loading models...")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.model, subfolder="scheduler"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model, subfolder="tokenizer"
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.model, subfolder="tokenizer_2"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model, subfolder="text_encoder"
        )
        self.text_encoder_2 = CLIPTextModel.from_pretrained(
            self.model, subfolder="text_encoder_2"
        )
        self.vae = AutoencoderKL.from_pretrained(self.model, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.model, subfolder="unet")
        self.unet.config.addition_embed_type = None
        # freeze parameters of models to save memory
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.to(self.device)
        self.unet.to(self.device)

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model, torch_dtype=self.weight_dtype
        )

    def load_data(self):
        print("loading data...")
        dataset = load_dataset(
            "imagefolder", data_dir=self.data_dir, keep_in_memory=True
        )
        dataset = dataset["train"]
        if self.max_training_data:
            dataset = dataset.select(range(self.max_training_data))
        self.dataset = dataset
        # print("encoding captions...")
        # self.dataset = self.dataset.map(lambda x: self.pipe.encode_prompt(x["caption"][0]))
        print(self.dataset)
        print("loading transforms, etc")
        self.dataloader = load_data(
            self.dataset,
            self.caption_column,
            self.image_column,
            self.resolution,
            self.center_crop,
            self.random_flip,
            self.train_batch_size,
            self.dataloader_num_workers,
            self.tokenizer,
            self.tokenizer_2,
            self.text_encoder,
            self.text_encoder_2,
        )
        print(self.dataloader)

    def training_prep(self):
        print("preparing for training...")
        # Freeze the unet parameters before adding adapters
        for param in self.unet.parameters():
            param.requires_grad_(False)

        unet_lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        # add adapter and make sure the trainable params are in float32.
        self.unet.add_adapter(unet_lora_config)
        lora_layers = filter(lambda p: p.requires_grad, self.unet.parameters())

        print("creating optimizer...")
        self.optimizer = torch.optim.AdamW(
            lora_layers,
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
        )

        # number of steps
        n_steps_per_epoch = math.ceil(len(self.dataloader) / self.grad_acc_steps)
        if self.max_train_steps is None:
            self.max_train_steps = self.num_train_epochs * n_steps_per_epoch

        self.lr_scheduler = get_scheduler(
            self.lr_scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.max_train_steps,
        )
        # afterwards we recalculate our number of training epochs
        num_train_epochs = math.ceil(self.max_train_steps / n_steps_per_epoch)
        total_batch_size = self.train_batch_size * self.grad_acc_steps

        print("***** Running training *****")
        print(f"  Num examples = {len(self.dataset)}")
        print(f"  Num Epochs = {num_train_epochs}")
        print(f"  Instantaneous batch size per device = {self.train_batch_size}")
        print(f"  Total train batch size = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {self.grad_acc_steps}")
        print(f"  Total optimization steps = {self.max_train_steps}")

    def training_loop(self):
        print("training...")
        for epoch in range(self.num_train_epochs):
            print(f"{epoch=}")
            self.unet.train()
            train_loss = 0.0
            for _, batch in enumerate(self.dataloader):
                # print(f"{step=}")
                loss = self.step(batch)

            train_loss += loss.item()
            logs = {
                "step_loss": train_loss,
                "lr": self.lr_scheduler.get_last_lr()[0],
            }
            print(logs)
            if self.wandb:
                self.wandb.log(logs)
            # self.progress_bar.set_postfix(logs)
            # if self.global_step >= self.max_train_steps:
            #     break

    # TODO this part is really complicated
    # mypy is having trouble with this method
    @typing.no_type_check
    def step(self, batch: dict[str, Tensor]):
        print(batch)
        for _, val in batch.items():
            val.to(self.device)
        pixels = batch["pixel_values"]
        pixels = pixels.to(dtype=self.weight_dtype)
        pixels = pixels.to(self.device)
        print(f"{pixels=}")
        print(f"{pixels.shape=}")
        vae_encoded = self.vae.encode(pixels)
        # print(f"{vae_encoded.shape=}")
        latents = vae_encoded.latent_dist.sample()
        print(f"{latents.shape=}")
        latents = latents * self.vae.config.scaling_factor

        # sample noise that we'll add to the latents
        noise = torch.randn_like(latents, device=self.device)
        print(f"{noise.shape=}")
        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1),
                device=latents.device,
            )
        bsz = latents.shape[0]
        print(f"{bsz=}")
        # sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()
        print(f"{timesteps=}")

        # add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        print(f"{noisy_latents.shape=}")

        # get the text embedding for conditioning
        enc_hidden_states = batch["encoding"]
        enc_hidden_states = enc_hidden_states.to(self.device)
        # enc_hidden_states = self.pipe.encode_prompt(batch["caption"])[0]
        # enc_hidden_states = self.text_encoder(batch["input_ids"], return_dict=False)[0]
        # enc_hidden_states = enc_hidden_states.to(self.device)
        print(f"{enc_hidden_states.shape=}")
        # get the target for loss depending on the prediction type
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.noise_scheduler.register_to_config(
                prediction_type=self.prediction_type
            )

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )
        print(f"{self.device=}")
        print(f"{noisy_latents.device=}")
        print(f"{enc_hidden_states.device=}")
        print(f"{timesteps=}")
        print(f"{self.unet.device=}")
        # predict the noise residual and compute loss
        model_pred = self.unet(
            noisy_latents, timesteps, enc_hidden_states, return_dict=False
        )[0]

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of
            # https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation
            # is slightly changed.  This is discussed in Section 4.2 of the same
            # paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values
                # before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack(
                    [snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                / snr
            )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return loss

    def save_model(self):
        """Save the Lora to `self.output_dir`."""
        print(f"saving model to: {self.output_dir}")
        self.unet = self.unet.to(torch.float32)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(self.unet)
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=self.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

    # TODO - this doesn't have to be part of the trainer...
    def validation(self):
        print("validating...")
        # generate some images to see if it worked...
        pipe = DiffusionPipeline.from_pretrained(
            self.model, torch_dtype=self.weight_dtype
        )
        pipe.safety_checker = safety_checker
        pipe.to(self.device)

        prompts = self.dataset["caption"]
        print(prompts)

        print("generating 'before' images...")
        for i, prompt in enumerate(prompts):
            gen = torch.Generator(device=self.device).manual_seed(0)
            images = pipe(prompt, num_images_per_prompt=2, generator=gen).images
            for j, image in enumerate(images):
                image.save(self.output_dir / f"val_before_{i}_{j}.png")

        print("generating 'after' images...")
        pipe.load_lora_weights(self.output_dir)
        pipe.fuse_lora(lora_scale=1.0)
        for i, prompt in enumerate(prompts):
            gen = torch.Generator(device=self.device).manual_seed(0)
            images = pipe(prompt, num_images_per_prompt=2, generator=gen).images
            for j, image in enumerate(images):
                image.save(self.output_dir / f"val_after_{i}_{j}.png")


def get_tokenize_function(tokenizer, caption_column):
    def tokenize(examples) -> Tensor:
        """Tokenize the captions of the given data."""
        inputs = tokenizer(
            examples[caption_column],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    return tokenize


def get_transforms(resolution, center_crop, random_flip) -> tv_transforms.Compose:
    """Defines the transformations used for training."""
    return tv_transforms.Compose(
        [
            tv_transforms.Resize(
                resolution,
                interpolation=tv_transforms.InterpolationMode.BILINEAR,
            ),
            (
                tv_transforms.CenterCrop(resolution)
                if center_crop
                else tv_transforms.RandomCrop(resolution)
            ),
            (
                tv_transforms.RandomHorizontalFlip()
                if random_flip
                else tv_transforms.Lambda(lambda x: x)
            ),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize([0.5], [0.5]),
        ]
    )


# pylint: disable-next=too-many-arguments
def load_data(
    dataset,
    caption_column,
    image_column,
    resolution,
    center_crop,
    random_flip,
    train_batch_size,
    dataloader_num_workers,
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
):
    print("encoding...")
    # encodings = list(map(lambda x: encoder_func(x)[0].detach().numpy(), dataset[caption_column]))
    # print(encodings)
    # dataset.add_column("encodings", encodings)
    # print(encodings)
    # tokenize_function = get_tokenize_function(tokenizer, caption_column)
    transforms = get_transforms(resolution, center_crop, random_flip)

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [transforms(image) for image in images]
        examples["encoding"] = encode_prompt(
            tokenizer,
            tokenizer_2,
            text_encoder,
            text_encoder_2,
            examples[caption_column],
        )
        return examples

    print("transforming data and tokenizing...", flush=True)
    dataset = dataset.with_transform(preprocess_train)
    print("finished transform.", flush=True)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=dataloader_num_workers,
    )
    return dataloader


def collate_fn(examples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    pixels = torch.stack([example["pixel_values"] for example in examples])
    pixels = pixels.to(memory_format=torch.contiguous_format).float()
    encoding = torch.stack([example["encoding"] for example in examples])
    return {"pixel_values": pixels, "encoding": encoding}
    # return {"pixel_values": pixels}


# pylint: disable-all
@typing.no_type_check
def encode_prompt(
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    prompt: str,
    prompt_2: Optional[str] = None,
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = True,
    negative_prompt: Optional[str] = None,
    negative_prompt_2: Optional[str] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    clip_skip: Optional[int] = None,
):
    # device = device or self._execution_device

    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # Define tokenizers and text encoders
    tokenizers = [tokenizer, tokenizer_2] if tokenizer is not None else [tokenizer_2]
    text_encoders = (
        [text_encoder, text_encoder_2] if text_encoder is not None else [text_encoder_2]
    )

    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # textual inversion: process multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):

            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                )
                print(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = text_encoder(
                text_input_ids.to(device), output_hidden_states=True
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            if clip_skip is None:
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    # get unconditional embeddings for classifier free guidance
    # zero_out_negative_prompt = (
    #     negative_prompt is None and self.config.force_zeros_for_empty_prompt
    # )
    zero_out_negative_prompt = True
    if (
        do_classifier_free_guidance
        and negative_prompt_embeds is None
        and zero_out_negative_prompt
    ):
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    elif do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt_2 = negative_prompt_2 or negative_prompt

        # normalize str to list
        negative_prompt = (
            batch_size * [negative_prompt]
            if isinstance(negative_prompt, str)
            else negative_prompt
        )
        negative_prompt_2 = (
            batch_size * [negative_prompt_2]
            if isinstance(negative_prompt_2, str)
            else negative_prompt_2
        )

        uncond_tokens: list[str]
        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = [negative_prompt, negative_prompt_2]

        negative_prompt_embeds_list = []
        for negative_prompt, tokenizer, text_encoder in zip(
            uncond_tokens, tokenizers, text_encoders
        ):

            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

            negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

    # if text_encoder_2 is not None:
    prompt_embeds = prompt_embeds.to(dtype=text_encoder_2.dtype, device=device)
    # else:
    #     prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        # if self.text_encoder_2 is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(
            dtype=text_encoder_2.dtype, device=device
        )
        # else:
        #     negative_prompt_embeds = negative_prompt_embeds.to(dtype=unet.dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(
            1, num_images_per_prompt, 1
        )
        negative_prompt_embeds = negative_prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

    # pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
    #     bs_embed * num_images_per_prompt, -1
    # )
    # if do_classifier_free_guidance:
    #     negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
    #         1, num_images_per_prompt
    #     ).view(bs_embed * num_images_per_prompt, -1)

    # return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    return prompt_embeds


# pylint: disable-next=unused-argument
def safety_checker(images, **kwargs):
    return images, [False] * len(images)


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    cli()
