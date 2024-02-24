# Mukh-Oboyob
Official Implementation of the IJACSA paper : Mukh-Oboyob: Stable Diffusion and BanglaBERT enhanced Bangla Text-to-Face Synthesis

![image](https://github.com/Codernob/Mukh-Oboyob/assets/55651740/2184cf28-6cd2-4b60-aaca-73ed1ae95c04)


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

## Inference using Huggingface
Alternatively you can use huggingface, pretrained model is uploaded to [Hugging Face](https://huggingface.co/ahmedfaiyaz/Mukh-Oboyob).
To run from Huggingface, use the code snippet below.
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
Example google colab notebook : https://colab.research.google.com/drive/1QGoaVVr89htsOx4jndKC_89IQhd7ywmT?usp=sharing

## Training
Extract `metadata.jsonl` from `Mukh-Oboyob/dataset/celeba/train/metadata.zip` and place on `Mukh-Oboyob/dataset/celeba/train/`
Now go to `Mukh-Oboyob/training script/` and run `train_text_to_image_lora.py`. 

## Citation
If you use my code or ideas from my paper in your work, please cite my paper.

```
@article{Saha2023,
title = {Mukh-Oboyob: Stable Diffusion and BanglaBERT enhanced Bangla Text-to-Face Synthesis},
journal = {International Journal of Advanced Computer Science and Applications},
doi = {10.14569/IJACSA.2023.01411142},
url = {http://dx.doi.org/10.14569/IJACSA.2023.01411142},
year = {2023},
publisher = {The Science and Information Organization},
volume = {14},
number = {11},
author = {Aloke Kumar Saha and Noor Mairukh Khan Arnob and Nakiba Nuren Rahman and Maria Haque and Shah Murtaza Rashid Al Masud and Rashik Rahman}
}
```

