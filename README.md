# Mukh-Oboyob
Official Implementation of the IJACSA paper : Mukh-Oboyob: Stable Diffusion and BanglaBERT enhanced Bangla Text-to-Face Synthesis

If you use my code in your work, please cite my paper.

```
Aloke Kumar Saha, Noor Mairukh Khan Arnob, Nakiba Nuren Rahman, Maria Haque, Shah Murtaza Rashid Al Masud and Rashik Rahman,
“Mukh-Oboyob: Stable Diffusion and BanglaBERT enhanced Bangla Text-to-Face Synthesis”
International Journal of Advanced Computer Science and Applications(IJACSA), 14(11), 2023.
http://dx.doi.org/10.14569/IJACSA.2023.01411142
```
# Usage
LoRA trained model is uploaded to [Hugging Face](https://huggingface.co/gr33nr1ng3r/Mukh-Oboyob).
To use that
```py
from diffusers import DiffusionPipeline
device="cuda"
pipeline = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
   
     custom_pipeline="gr33nr1ng3r/Mukh-Oboyob"
)
pipeline.unet.load_attn_procs("gr33nr1ng3r/Mukh-Oboyob")
pipeline.to(device)
prompt = "মেয়েটির কালো চুল ছিল। মেয়েটির মুখে ভারী মেকাপ ছিল। মেয়েটির উঁচু গালের হাড় ছিল। মেয়েটির মুখ কিছুটা খোলা ছিল। মেয়েটির চেহারা ডিম্বাকৃতির। মেয়েটির চোখা নাক ছিল। মেয়েটির ঢেউ খেলানো চুল ছিল। মেয়েটির কানে দুল পরা ছিল। মেয়েটির লিপস্টিক পরা ছিল। "
image = pipeline(prompt, num_inference_steps=200, guidance_scale=7.5,height=128,width=128).images[0]
image

```
