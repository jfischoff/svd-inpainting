import torch

from diffusers.utils import load_image, export_to_video
from PIL import Image
from einops import rearrange
import os
import ffmpeg
import os
import torch
import datetime
import numpy as np
from PIL import Image
from svd_inpainting.pipeline.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionPipelineControlNet
from svd_inpainting.models.controlnet_sdv import ControlNetSDVModel
from svd_inpainting.models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
import cv2
import re

def tensor_to_pil(tensor):
    """ Convert a PyTorch tensor to a PIL Image. """
    # Convert tensor to numpy array
    if len(tensor.shape) == 4:  # batch of images
        images = [Image.fromarray(img.numpy().transpose(1, 2, 0)) for img in tensor]
    else:  # single image
        images = Image.fromarray(tensor.numpy().transpose(1, 2, 0))
    return images
# Define functions
def validate_and_convert_image(image, target_size=(256, 256)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        # Resize PIL Image
        image = image.resize(target_size)
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None

    return image

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

def load_images_from_folder_to_pil(folder, target_size=(512, 512)):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    def frame_number(filename):
        matches = re.findall(r'\d+', filename)  # Find all sequences of digits in the filename
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'


    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)

    # Load, resize, and convert images
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read image with original channels
            if img is not None:
                # Resize image
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                # Convert to uint8 if necessary
                if img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)

                # Ensure all images are in RGB format
                if len(img.shape) == 2:  # Grayscale image
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image in BGR format
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Convert the numpy array to a PIL image
                pil_img = Image.fromarray(img)
                images.append(pil_img)

    return images

base_model = "stabilityai/stable-video-diffusion-img2vid-xt"
controlnet = controlnet = ControlNetSDVModel.from_pretrained(
    "CiaraRowles/temporal-controlnet-depth-svd-v1",
    subfolder="controlnet",
    torch_dtype=torch.float16,
    )
unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
    base_model,
    subfolder="unet",
    torch_dtype=torch.float16,
    variant="fp16",
    )
pipe = StableVideoDiffusionPipelineControlNet.from_pretrained(
    base_model,
    unet=unet,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda:0")


# Load the conditioning image
# image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
# image = Image.open("assets/rocket.png")
image = Image.open("validation_images/chair.png")
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

validation_control_images = load_images_from_folder_to_pil("validation_images/depth")


generator = torch.manual_seed(42)
frames = pipe(image,
              mask_image=None,
              add_predicted_noise=False,
              decode_chunk_size=1,
              generator=generator,
              num_inference_steps=20,
              controlnet_condition=validation_control_images[:14],
              num_frames=14,
              callback_on_step_end=None).frames[0]

save_images(frames, "output/quay")
frames_to_video("output/quay", "output/quay.mp4")
