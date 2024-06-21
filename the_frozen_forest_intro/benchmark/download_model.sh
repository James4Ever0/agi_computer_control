export HF_ENDPOINT=https://hf-mirror.com
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

mkdir -p $MODEL_NAME
env http_proxy="" https_proxy="" all_proxy="" ./hfd.sh $MODEL_NAME --tool aria2c --local-dir $MODEL_NAME --include '*.safetensors'