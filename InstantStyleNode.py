from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
current_directory = os.path.dirname(os.path.abspath(__file__))

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    image_np = (255. * input_image.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)
    input_image = Image.fromarray(image_np)

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

class PromptLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "a cat, best quality", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly", "multiline": True}),
            }
        }

    RETURN_TYPES = ('STRING','STRING',)
    RETURN_NAMES = ('positive_prompt','negative_prompt',)
    FUNCTION = "prompt"
    CATEGORY = "InstantStyle"

    def prompt(self, prompt, negative_prompt):
        return prompt, negative_prompt

class BaseModelLoader:
    """
    A simple base model loader node
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_path": ("STRING", { "default": "checkpoints/realvisxlV40_v40Bakedvae.safetensors"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "InstantStyle"
  
    def load_model(self, ckpt_path):
                
        pipe = StableDiffusionXLPipeline.from_single_file(
            pretrained_model_link_or_path=ckpt_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(device)
        return [pipe]


class InstantStyleLoader:
    """
    A simple instantstyle loader node
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter_path": ("STRING", {"default": "checkpoints/IP-Adapter"}),
                "subfolder": ("STRING", {"default": "sdxl_models"}),
                "image_encoder_folder": ("STRING", {"default": "image_encoder"}),
                "filename": ("STRING", {"default": "ip-adapter_sdxl.bin"}),
                "pipe": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_instantstyle"
    CATEGORY = "InstantStyle"

    def load_instantstyle(self, pipe, ipadapter_path, filename, subfolder, image_encoder_folder):

        # load ip-adapter
        pipe.load_ip_adapter(ipadapter_path, 
                             subfolder=subfolder, weight_name=filename, image_encoder_folder=image_encoder_folder)

        # configure ip-adapter scales.
        scale = {
            "down": {"block_2": [0.0, 1.0]},
            "up": {"block_0": [0.0, 1.0, 0.0]},
        }
        pipe.set_ip_adapter_scale(scale)

        return [pipe]

class InstantStyleGenerationNode:
    """
    A simple instantstyle inference node
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": ("IMAGE",),
                "pipe": ("MODEL",),
                "positive": ("STRING", {"multiline": True}),
                "negative": ("STRING", {"multiline": True}),
                "mode": (["style", "layout", "style+layout", "ip-adapter"],),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "guidance_scale": ("FLOAT", {"default": 5, "min": 0, "max": 10, "display": "slider"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "InstantStyle"
                       
    def generate_image(self, positive, negative, style_image, pipe, mode, steps, guidance_scale, seed):
        
        generator = torch.Generator(device=device).manual_seed(seed)
        
        style_image = resize_img(style_image)        
        
        if mode == "style":
            scale = {
                "down": {"block_2": [0.0, 0.0]},
                "up": {"block_0": [0.0, 1.0, 0.0]},
            }
        elif mode == "layout":
            scale = {
                "down": {"block_2": [0.0, 1.0]},
                "up": {"block_0": [0.0, 0.0, 0.0]},
            }
        elif mode == "style+layout":
            scale = {
                "down": {"block_2": [0.0, 1.0]},
                "up": {"block_0": [0.0, 1.0, 0.0]},
            }
        else:
            scale = 1.0
            
        pipe.set_ip_adapter_scale(scale)

        output = pipe(
            prompt=positive,
            negative_prompt=negative,
            ip_adapter_image=style_image,
            num_inference_steps=steps,
            generator=generator,
            guidance_scale=guidance_scale,
        )

        if isinstance(output, tuple):
            images_list = output[0]
        else:
            images_list = output.images

        images_tensors = []
        for img in images_list:
            img_array = np.array(img)
            img_tensor = torch.from_numpy(img_array).float() / 255.
            if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
            images_tensors.append(img_tensor)

        if len(images_tensors) > 1:
            output_image = torch.cat(images_tensors, dim=0)
        else:
            output_image = images_tensors[0]

        return (output_image,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "PromptLoader": PromptLoader,
    "BaseModelLoader": BaseModelLoader,
    "InstantStyleLoader": InstantStyleLoader,
    "InstantStyleGenerationNode": InstantStyleGenerationNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptLoader": "PromptLoader",
    "BaseModelLoader": "BaseModelLoader",
    "InstantStyleLoader": "InstantStyleLoader",
    "InstantStyleGenerationNode": "InstantStyleGenerationNode",
}
