import os
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

hf_token = os.getenv("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")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=hf_token
).to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="Bir metin gir..."),
    outputs="image"
)

if __name__ == "__main__":
    demo.launch()