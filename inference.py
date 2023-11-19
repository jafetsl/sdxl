from diffusers import StableDiffusionPipeline
import torch
import os


pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.load_lora_weights("/home/jafet/Desktop/sd_db_lora/output/pytorch_lora_weights.safetensors")

prompt = "A photo of sks dog in a bucket"
image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("dog-bucket.png")