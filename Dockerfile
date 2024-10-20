# Base image
FROM runpod/base:0.4.2-cuda11.8.0

ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Install additional dependencies for LoRA and safetensors
RUN python3.11 -m pip install huggingface_hub

# Cache Models
COPY builder/cache_models.py /cache_models.py
RUN python3.11 /cache_models.py && \
    rm /cache_models.py

# Optional: Download Civitai model (Replace with actual URL or path)
# This section is optional if you're downloading the model directly in code
# RUN wget -O /models/ponydiffusion6.safetensors "https://civitai.com/api/download/models/<model_id>"

# Add src files (Worker Template)
ADD src .

CMD python3.11 -u /rp_handler.py
