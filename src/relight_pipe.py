import os
import torch
import imageio
import argparse
import numpy as np
import safetensors.torch as sf
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.hub import download_url_to_file

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKLWan, AutoencoderKL, UNet2DConditionModel
from diffusers import DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.models.attention_processor import AttnProcessor2_0

from utils.tools import set_all_seed, read_video
from src.ic_light_pipe import StableDiffusionImg2ImgPipeline
from src.ic_light import BGSource
from pipleline import WanVideoRelightPipeline

def main(args):

    config = OmegaConf.load(args.config)
    device = torch.device('cuda')
    adopted_dtype = torch.float16
    set_all_seed(42)
    vdm_model_path = './models/wan_vdm'
    sd_model_path = './models/stabel_diffusion'

    ## vdm model        
    vae = AutoencoderKLWan.from_pretrained(args.vdm_model, subfolder="vae", torch_dtype=adopted_dtype, cache_dir=vdm_model_path)
    pipe = WanVideoRelightPipeline.from_pretrained(args.vdm_model, vae=vae, torch_dtype=adopted_dtype, cache_dir=vdm_model_path)
    FlowMatching_scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    pipe.scheduler = FlowMatching_scheduler

    pipe = pipe.to(device=device, dtype=adopted_dtype)
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    ## module
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer", cache_dir=sd_model_path)
    text_encoder = CLIPTextModel.from_pretrained(args.sd_model, subfolder="text_encoder", cache_dir=sd_model_path)
    vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae", cache_dir=sd_model_path)
    unet = UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet", cache_dir=sd_model_path)
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in
    unet_original_forward = unet.forward

    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
        # c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        if c_concat.shape[0] != sample.shape[0]:
            repeat_factor = -(-sample.shape[0] // c_concat.shape[0])
            c_concat = torch.cat([c_concat] * repeat_factor, dim=0)
            c_concat = c_concat[:sample.shape[0]]
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs['cross_attention_kwargs'] = {}
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)
    unet.forward = hooked_unet_forward

    ## ic-light model loader
    
if __name__ == "main":

    parser = argparse.ArgumentParser()

    parser.add_argument("--sd_model", type=str, default="stablediffusionapi/realistic-vision-v51")
    parser.add_argument("--vdm_model", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--ic_light_model", type=str, default="./models/iclight_sd15_fc.safetensors")
    parser.add_argument("--config", type=str, default="configs/wan_relight/man.yaml", help="the config file for each sample.")

    args = parser.parse_args()
    main(args)

    