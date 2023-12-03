import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=torch.float16,
)

model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
scheduler = DDIMScheduler.from_pretrained(
    model_id, 
    subfolder="scheduler", 
    clip_sample=False, 
    timestep_spacing="linspace", 
    steps_offset=1,
    torch_dtype=torch.float16,
)

pipe = AnimateDiffPipeline.from_pretrained(
    model_id, 
    motion_adapter=adapter, 
    torch_dtype=torch.float16, 
    use_safetensors=True
)
pipe.scheduler = scheduler

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

prompts = ["masterpiece, bestquality, highlydetailed, ultradetailed, hyper realistic, sunset, "]
n_prompts = ["bad quality, worse quality"]
generators = [torch.Generator().manual_seed(42)]

with torch.inference_mode():
    output = pipe(
        prompt=prompts,
        negative_prompt=n_prompts,
        num_frames=16,
        guidance_scale=7.5,
        num_inference_steps=25,
        generator=generators,
    )


frames = output.frames[0]
export_to_gif(frames, "animation.gif")