import time
from threading import Thread
import torch
import numpy as np
import cv2
from queue import Queue
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from pytorch_unet import UNet, SRUnet, SimpleResNet
import data_loader as dl
import utils
from models import *

# Enable cuDNN benchmark mode for optimized performance
torch.backends.cudnn.benchmark = True


# Function to calculate padded dimensions
def get_padded_dim(H_x, W_x, border=0, mod=16):
    padH = ((mod + border) - H_x % (mod + border)) % (mod + border)
    padW = ((mod + border) - W_x % (mod + border)) % (mod + border)
    return H_x + padH, W_x + padW, padH, padW


# Function to configure the generator model based on the architecture name
def configure_generator(arch_name, dataset_upscale_factor, args):
    if arch_name == 'srunet':
        model = SRUnet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS,
                       downsample=args.DOWNSAMPLE, layer_multiplier=args.LAYER_MULTIPLIER)
    elif arch_name == 'unet':
        model = UNet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS)
    elif arch_name == 'srgan':
        model = SRResNet()
    elif arch_name == 'espcn':
        model = SimpleResNet(n_filters=64, n_blocks=6)
    else:
        model = None
    if model is None:
        raise Exception("Unknown architecture. Select one between:", args.archs)

    print("Loading model: ", args.MODEL_NAME)
    model.load_state_dict(torch.load(args.MODEL_NAME, weights_only=True))
    return model


# Function to read and preprocess frames from the video
def read_pic(reader, q, padH, padW, args):
    while True:
        try:
            cv2_im = next(reader)['data'].cuda().float()
            x = dl.normalize_img(cv2_im / 255.).unsqueeze(0)
            x_bicubic = torch.clip(F.interpolate(x, scale_factor=args.UPSCALE_FACTOR * args.DOWNSAMPLE, mode='bicubic'), min=-1, max=1)
            x = F.pad(x, [0, padW, 0, padH])
            q.put((x, x_bicubic))
        except StopIteration:
            print("End of video stream.")
            break
        except Exception as e:
            print(f"Error reading frame: {e}")
            break


if __name__ == '__main__':
    # Parse arguments and set device
    args = utils.ARArgs()
    device = torch.device(f"cuda:{args.CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # Configure and load the generator model
    model = configure_generator(args.ARCHITECTURE, args.UPSCALE_FACTOR, args)
    model.to(device)
    model.reparametrize()
    model.eval()

    # Open video file and get video properties
    cap = cv2.VideoCapture(args.CLIPNAME)
    reader = torchvision.io.VideoReader(args.CLIPNAME, "video")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_fix, width_fix, padH, padW = get_padded_dim(height, width)

    # Initialize frame and output queues
    frame_queue = Queue(1)
    out_queue = Queue(1)
    reader.seek(0)

    # Start thread to read and preprocess frames
    Thread(target=read_pic, args=(reader, frame_queue, padH, padW, args)).start()

    target_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frametime = 0

    # Process frames with the model
    with torch.no_grad():
        for _ in tqdm(range(frame_count), desc="Processing frames"):
            t0 = time.time()
            x, x_bicubic = frame_queue.get()
            out = model(x)[:, :, :height * 2, :width * 2]
            frametime = time.time() - t0
            total_frametime += frametime
            tqdm.write(f"Frame time: {frametime * 1e3:.2f} ms; FPS: {1 / frametime:.2f}")

    # Calculate and print average frame time and FPS
    avg_frametime = total_frametime / frame_count
    avg_fps = 1 / avg_frametime
    print(f"Average frame time: {avg_frametime * 1e3:.2f} ms; Average FPS: {avg_fps:.2f}")
