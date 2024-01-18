# Mukh-Oboyob
Official Implementation of the IJACSA paper : Mukh-Oboyob: Stable Diffusion and BanglaBERT enhanced Bangla Text-to-Face Synthesis

## Environment Setup
Run the following commands on your anaconda environment.
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.22.0
pip install transformers==4.29.2
pip install matplotlib==3.7.2
```

## Library Modifications

1. I modified the stable diffusion pipeline to load the `BanglaBERT`` text encoder along with setting the max sequence length to 150 (reason explained in my paper). So you have to take the file from
```
C:\Users\USER_NAME\anaconda3\envs\ENVIRONMENT_NAME\Lib\site-packages\diffusers\pipelines\stable_diffusion\pipeline_stable_diffusion.py
```
and replace it with the file from this github repository
```
Codernob\Mukh-Oboyob\library codes\pipeline_stable_diffusion.py
```

Don't forget to replace `USER_NAME` with the user name in which your anaconda distribution is installed and replace `ENVIRONMENT_NAME` with your intended anaconda environment.

2. The BanglaBERT text encoder is not supported natively by the stable diffusion pipeline, as it expects a CLIP text encoder by default. So I had to comment out some code that checks for CLIP in pipeline_utils.py. Therefore, replace the file from 
```
C:\Users\USER_NAME\anaconda3\envs\ENVIRONMENT_NAME\Lib\site-packages\diffusers\pipelines\pipeline_utils.py
```
and replace it with the file from my github repository
```
Codernob\Mukh-Oboyob\library codes\pipeline_utils.py
```

3. From my experience, the safety checker gives many false positives. So I turned it off. If you want to do so, replace this file
```
C:\Users\USER_NAME\anaconda3\envs\ENVIRONMENT_NAME\Lib\site-packages\diffusers\pipelines\stable_diffusion\safety_checker.py
```
with
```
Codernob\Mukh-Oboyob\library codes\safety_checker.py
```

## Inference

Now you should be able to run `Codernob\Mukh-Oboyob\inference\sample inference.ipynb`

## Citation
If you use my code in your work, please cite my paper.

```
Aloke Kumar Saha, Noor Mairukh Khan Arnob, Nakiba Nuren Rahman, Maria Haque, Shah Murtaza Rashid Al Masud and Rashik Rahman,
“Mukh-Oboyob: Stable Diffusion and BanglaBERT enhanced Bangla Text-to-Face Synthesis”
International Journal of Advanced Computer Science and Applications(IJACSA), 14(11), 2023.
http://dx.doi.org/10.14569/IJACSA.2023.01411142
```
<!-- # Usage
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

``` -->
