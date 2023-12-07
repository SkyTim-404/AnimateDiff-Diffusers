import torch
from diffusers import (
    MotionAdapter, 
    AnimateDiffPipeline, 
    StableDiffusionPipeline, 
    DDIMScheduler, 
    LMSDiscreteScheduler, 
    DPMSolverMultistepScheduler, 
    LCMScheduler,
    UNet2DConditionModel, 
    UNetMotionModel, 
)
from diffusers.utils import export_to_gif
from omegaconf import OmegaConf
import datetime
import os
from pathlib import Path
import random

def main(config_path):
    config  = OmegaConf.load(config_path)
    
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = f"samples/{Path(config_path).stem}-{time_str}"
        
        adapter = model_config.get("adapter", "guoyww/animatediff-motion-adapter-v1-5-2")
        checkpoint = model_config.get("checkpoint", "models/Checkpoint/CounterfeitV30.safetensors")
        sd_model = model_config.get("sd_model", "SG161222/Realistic_Vision_V5.1_noVAE")
        lora = model_config.get("lora", [])
        textual_inversion = model_config.get("textual_inversion", [])
        use_lcm = model_config.get("use_lcm", False)
        lcm_path = model_config.get("lcm_path", "")
        
        adapter = MotionAdapter.from_pretrained(
            adapter,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        if checkpoint != "":
            checkpoint = StableDiffusionPipeline.from_single_file(
                checkpoint, 
                torch_dtype=torch.float16, 
                use_safetensors=True
            )
            
            unet = UNetMotionModel.from_unet2d(checkpoint.unet, adapter)
            vae = checkpoint.vae
            feature_extractor = checkpoint.feature_extractor
            text_encoder = checkpoint.text_encoder
            tokenizer = checkpoint.tokenizer
            scheduler = checkpoint.scheduler
            
            pipe = AnimateDiffPipeline(
                vae=vae, 
                text_encoder=text_encoder, 
                tokenizer=tokenizer, 
                feature_extractor=feature_extractor, 
                unet=unet, 
                scheduler=scheduler,
                motion_adapter=adapter, 
            )
        else:
            pipe = AnimateDiffPipeline.from_pretrained(
                sd_model, 
                motion_adapter=adapter, 
                torch_dtype=torch.float16, 
                use_safetensors=True, 
            )
            
        pipe = pipe.to(dtype=torch.float16)
            
        if use_lcm:
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights(lcm_path)
        
        for l in lora:
            print(f"loading lora weights from {l}")
            pipe.load_lora_weights(l)
            
        for ti in textual_inversion:
            print(f"loading textual inversion weights from {ti}")
            pipe.load_textual_inversion(ti)
            

        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()

        repeat = model_config.get("repeat", 1)
        prompts = model_config.get("prompt", [""])
        n_prompts = model_config.get("n_prompt", ["worse quality, low quality"])
        random_seeds = model_config.get("seed", [0])
        steps = model_config.get("steps", [25])
        guidance_scales = model_config.get("guidescale", [7.5])
        
        n_prompts = list(n_prompts) * len(prompts) if len(n_prompts) == 1 else n_prompts
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
        steps = [steps] if isinstance(steps, int) else list(steps)
        steps = steps * len(prompts) if len(steps) == 1 else steps
        guidance_scales = [guidance_scales] if isinstance(guidance_scales, float) else list(guidance_scales)
        guidance_scales = guidance_scales * len(prompts) if len(guidance_scales) == 1 else guidance_scales

        os.makedirs(savedir)
        
        for i in range(repeat):
            generators = [torch.Generator().manual_seed(seed) if seed != -1 else torch.Generator().manual_seed(random.randint(0, 10000000000)) for seed in random_seeds]
            for prompt_idx, (prompt, n_prompt, generator, step, guidance_scale) in enumerate(zip(prompts, n_prompts, generators, steps, guidance_scales)):
                print(f"sampling {prompt} ...")
                with torch.inference_mode():
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=n_prompt,
                        height=512,
                        width=512,
                        num_frames=16,
                        guidance_scale=guidance_scale,
                        num_inference_steps=step,
                        generator=generator,
                    )


                frames = output.frames[0]
                os.makedirs(f"{savedir}/sample", exist_ok=True)
                export_to_gif(frames, f"{savedir}/sample/{i}-{prompt_idx}.gif")
            
            model_config["seed"] = [g.seed() for g in generators]
            OmegaConf.save(model_config, f"{savedir}/config.yaml")
    
    
if __name__ == "__main__":
    config_path = "configs/test_elegent.yaml"
    main(config_path)