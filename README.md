# ComfyUI-InstantStyle

Personal toy implementation of [InstantStyle](https://github.com/InstantStyle/InstantStyle) for ComfyUI. For complete node support, refer to [here](https://github.com/cubiq/ComfyUI_IPAdapter_plus).

<img src="https://github.com/haofanwang/ComfyUI-InstantStyle/blob/main/example/workflow.png?raw=true" width="100%" height="100%">

## How to use

Prepare
```
cd custom_nodes
git clone https://github.com/haofanwang/ComfyUI-InstantStyle.git
cd custom_nodes/ComfyUI-InstantStyle
pip install -r requirements.txt
```

Download model

```
huggingface-cli download --resume-download h94/IP-Adapter --local-dir checkpoints/IP-Adapter
```

If you cannot access to HuggingFace,
```
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download h94/IP-Adapter --local-dir checkpoints/IP-Adapter
```

Then, download any base model and save it as `checkpoints/realvisxlV40_v40Bakedvae.safetensors`.

## Acknowledgement
This project is highly inspired by [ZHO-ZHO-ZHO](https://github.com/ZHO-ZHO-ZHO).
