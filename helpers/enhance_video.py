from queue import Queue
from threading import Thread
from tqdm import tqdm
from pathlib import Path
from models import *
from pytorch_unet import *
from render import cv2toTorch, torchToCv2

import cv2
import utils, os
import data_loader as dl
import numpy as np

args = utils.ARArgs()


def cat_dim(t1, t2):
    return torch.cat([t1, t2], dim=1)


def save_with_cv(pic, imname):
    pic = dl.de_normalize(pic.squeeze(0))
    npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0)) * 255
    npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)

    cv2.imwrite(imname, npimg)


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
        raise Exception("Unknown architecture. Select one between:", args.archs)

    print("Loading model: ", filename)
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict)

    return model


def enhance_video(test_dir_prefix, video_prefix, filename, from_second=0, to_second=None, crf=None):
    device = torch.device(f"cuda:{args.CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")

    ### Load model
    arch_name = args.ARCHITECTURE
    dataset_upscale_factor = args.UPSCALE_FACTOR
    model = configure_generator(arch_name, dataset_upscale_factor, args)
    model.to(device)
    model.eval()

    ### Load video 
    resolution_lq = args.TEST_INPUT_RES
    crf_ = crf if crf is not None else 22
    print(f"Testing: {video_prefix} with CRF {crf_}")
    lq_file_path = str(test_dir_prefix) + f"/encoded{resolution_lq}CRF{crf_}/" + video_prefix + ".mp4"
    cap_lq = cv2.VideoCapture(lq_file_path)

    ## Queues
    lq_queue = Queue(1) 
    out_queue = Queue(1)

    ## Frame count
    total_frames = int(cap_lq.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap_lq.get(cv2.CAP_PROP_FPS))

    from_frame = fps * from_second
    to_frame = fps * to_second

    if to_frame is None:
        to_frame = total_frames

    to_frame = min(to_frame, total_frames)

    ## Read Thread
    def read_pic(cap, q, from_frame_, to_frame_):
        count = 0
        cap.set(cv2.CAP_PROP_POS_MSEC, from_second)
        while cap.isOpened():
            success, cv2_im = cap.read()
            if success:
                cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                x = cv2toTorch(cv2_im)
                q.put(x)
                count += 1
                if count == (to_frame_ - from_frame_):
                    print("Releasing cap")
                    cap.release()
            else:
                cap.release()

    ## Save Thread
    def save_pic(q):
        count = 0
        while True:
            imname = video_prefix + f"_{count}.png"

            frame_name_pattern = imname.split(".")[0].split("_")[:-1]
            frame_name_pattern = "_".join(frame_name_pattern) + "_frame"
            frame_name_pattern = dest / frame_name_pattern

            imname = str(dest / imname)
            y_fake = q.get()
            if y_fake is not None:
                save_with_cv(y_fake, imname)
                count += 1
            else:
                break
    
    ## Create output directory
    dest = test_dir_prefix.split("/")
    dest_dir = Path("/".join(dest))

    model_name = os.path.basename(os.path.dirname(os.path.normpath(filename)))
    res = "1.5x" if resolution_lq == 720 else "2x"

    dest = dest_dir / 'models' / f'{res}' / f'{crf}' / model_name / 'out' / video_prefix
    print("Output directory: ", dest)
    dest.mkdir(exist_ok=True, parents=True)

    clip_gen_folder = dest_dir / 'models' / f'{res}' / f'{crf}' / model_name 
    os.makedirs(clip_gen_folder, exist_ok=True)

    ## Get video resolution
    H_x = int(cap_lq.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W_x = int(cap_lq.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Src resolution: {W_x}x{H_x}")

    ## Pad video
    border = 0
    modH, modW = H_x % (16 + border), W_x % (16 + border)
    padW = ((16 + border) - modW) % (16 + border)
    padH = ((16 + border) - modH) % (16 + border)

    new_H = H_x + padH
    new_W = W_x + padW

    model.batch_size = 1
    model.width = new_W  # x.shape[-1] + (patch_size - modW) % patch_size
    model.height = new_H  # x.shape[-2] + (patch_size - modW) % patch_size
    print(f"Padded src resolution: {new_W}x{new_H}")

    ## Start threads
    thread1 = Thread(target=read_pic, args=(cap_lq, lq_queue, from_frame, to_frame))  # .start()
    thread3 = Thread(target=save_pic, args=(out_queue,))  # .start()
    thread1.start()
    thread3.start()

    ## Start testing
    for i in tqdm(range(to_frame - from_frame), total=to_frame - from_frame, desc=f"Testing {video_prefix}"):
        with torch.no_grad():
            # Get frames
            x = lq_queue.get().to(device)
            x = F.pad(x, [0, padW, 0, padH])
            # Enhance
            y_fake = model(x)
            out_queue.put(y_fake) 

    out_queue.put(None)

    ffmpeg_command = f"ffmpeg -nostats -loglevel 0 -framerate {fps} -start_number 0 -i\
        {dest}/{video_prefix}_%d.png -crf 5  -c:v libx265 -r {fps} \
        -pix_fmt yuv420p {clip_gen_folder / f'{video_prefix}.mp4 -y'}"
    
    print("Putting output images together.\n", ffmpeg_command)
    os.system(ffmpeg_command)

    print("Test completed")


if __name__ == '__main__':
    test_dir = Path(args.TEST_DIR)
    videos = [vid.strip(".y4m") for vid in os.listdir(test_dir) if vid.endswith('.y4m') and '1080' in vid]

    second_start = 0
    second_finish = 120  # test no more than the 2nd minutes - none of the test videos last so much

    for crf, filename in [(args.CRF, args.MODEL_NAME),]:
        print(f"Testing CRF {crf}, with model {filename}")

        for vid in tqdm(videos, desc=f"Enhancing Videos"):
            enhance_video(
                str(test_dir), video_prefix=vid, filename=filename,
                from_second=second_start, to_second=second_finish, crf=crf
            )
