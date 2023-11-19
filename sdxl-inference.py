from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("jafetsierra/path-to-save-model")

prompt = "A photo of sks dog in a bucket"
image = pipeline(prompt, num_inference_steps=10, guidance_scale=7.5).images[0]

image.save("dog-bucket.png")