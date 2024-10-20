# builder/model_fetcher.py
import gdown
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from peft import PeftModel
import os

lora_folder = "./loras"
os.makedirs(lora_folder, exist_ok=True)

# Функция для скачивания файла с Google Диска
def download_from_google_drive(file_id, output_path):
    google_drive_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(google_drive_url, output_path, quiet=False)

# Скачивание модели и LoRA с Google Диска
def download_model_and_lora():
    # Загрузка LoRA весов
    lora_file_id = '1GBKiXaKlt2n4IEmFp0NVf_4pzIf5gA8e'  # ID файла LoRA на Google Drive
    lora_output_path = os.path.join(lora_folder, 'fcomic.safetensors')
    if not os.path.exists(lora_output_path):
        download_from_google_drive(lora_file_id, lora_output_path)

    return lora_output_path


def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def get_diffusion_pipelines():
    '''
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    '''

    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True
    }

    pipe = fetch_pretrained_model(StableDiffusionXLPipeline,
                                  "LyliaEngine/Pony_Diffusion_V6_XL", **common_args)
    # Применяем LoRA веса, загруженные с Civitai
    lora_weights_path = download_model_and_lora()

    vae = fetch_pretrained_model(
        AutoencoderKL, "LyliaEngine/Pony_Diffusion_V6_XL", **{"torch_dtype": torch.float16}
    )
    print("Loaded VAE")
    refiner = fetch_pretrained_model(StableDiffusionXLImg2ImgPipeline,
                                     "stabilityai/stable-diffusion-xl-refiner-1.0", **common_args)

    # Применение LoRA
    pipe = PeftModel.from_pretrained(pipe, lora_weights_path)

    return pipe, refiner, vae


if __name__ == "__main__":
    get_diffusion_pipelines()
