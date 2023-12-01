import torch

from pipeline_stable_video_diffusion_inpaint import StableVideoDiffusionInpaintingPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image

pipe = StableVideoDiffusionInpaintingPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")


# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
image = image.resize((1024, 576))

mask_image = Image.open("mask.png")

mask_image = mask_image.resize((1024, 576))

def callback(pipe, i, j, dict):
  latents = dict["latents"]
  # decode the first frame latents and save it
  first_frame = latents[:, 0:1]

  decoded_first_frame = pipe.decode_latents(first_frame, 1)
  decoded_first_frame = decoded_first_frame.permute(0, 2, 3, 4, 1)[0][0]
  

  latent_path = f"output/latent_{str(i).zfill(4)}.png"

  decoded_first_frame = (decoded_first_frame + 1) * 127.5
  decoded_first_frame = decoded_first_frame.cpu().numpy().astype("uint8")
  decoded_first_frame = Image.fromarray(decoded_first_frame)
  decoded_first_frame.save(latent_path)


  return dict


generator = torch.manual_seed(42)
frames = pipe(image, 
              mask_image,
              add_predicted_noise=False,
              decode_chunk_size=1, 
              generator=generator, 
              num_inference_steps=50,
              callback_on_step_end=None).frames[0]

export_to_video(frames, "output/generated.mp4", fps=7)