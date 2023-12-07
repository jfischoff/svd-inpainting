import torch

from svd_inpainting.pipeline_stable_video_diffusion_inpaint import StableVideoDiffusionInpaintingPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
from einops import rearrange
import os
import ffmpeg

def save_images(frames, output_folder):
  os.makedirs(output_folder, exist_ok=True)
  for i, frame in enumerate(frames):
    frame.save(f"{output_folder}/{str(i).zfill(4)}.png")

def frames_to_video(input_folder,
                    output_file,
                    pattern='%04d.png',
                    frame_rate=8,
                    vcodec='libx264',
                    crf=18,
                    preset='veryslow',
                    pix_fmt='yuv420p'):
  # Define input file pattern
  input_pattern = os.path.join(input_folder, pattern)

  # Create FFmpeg input stream from image sequence
  input_stream = ffmpeg.input(input_pattern, framerate=frame_rate)

  # create the directory for the output file
  os.makedirs(os.path.dirname(output_file), exist_ok=True)

  output_stream = ffmpeg.output(input_stream,
                                output_file,
                                vcodec=vcodec,
                                crf=crf,
                                preset=preset,
                                pix_fmt=pix_fmt,
                                y='-y')
  # Run FFmpeg command to convert image sequence to video
  ffmpeg.run(output_stream)



pipe = StableVideoDiffusionInpaintingPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")


# Load the conditioning image
# image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
image = Image.open("assets/rocket.png")
image = image.resize((1024, 576))

mask_image = Image.open("assets/rocket_mask.png")
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
              num_inference_steps=100,
              callback_on_step_end=None).frames[0]

save_images(frames, "output/quay")
frames_to_video("output/quay", "output/quay.mp4")
