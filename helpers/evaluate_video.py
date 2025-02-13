import pandas as pd
import utils, os

args = utils.ARArgs()

from pathlib import Path

from tqdm import tqdm
import data_loader as dl
import pytorch_ssim as torch_ssim
import lpips
import numpy as np

from models import *
from pytorch_unet import *
from render import cv2toTorch, torchToCv2
import cv2
from queue import Queue
from threading import Thread
from vmaf_torch import VMAF


def cat_dim(t1, t2):
    return torch.cat([t1, t2], dim=1)


def save_with_cv(pic, imname):
    pic = dl.de_normalize(pic.squeeze(0))
    npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0)) * 255
    npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)

    cv2.imwrite(imname, npimg)


def evaluate_video(test_dir_prefix, video_prefix, from_second=0, to_second=None, crf=None):
    device = torch.device(f"cuda:{args.CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")

    ### Load all metrics
    lpips_metric = lpips.LPIPS(net='alex')
    vmaf_metric = VMAF(temporal_pooling=True, enable_motion=True)
    vmaf_neg_metric = VMAF(temporal_pooling=True, enable_motion=True, NEG=True)
    ssim = torch_ssim.SSIM(window_size=11)

    ### Move metrics to device
    lpips_metric.to(device)
    vmaf_metric.to(device)
    vmaf_neg_metric.to(device)
    ssim.to(device)
    # vmaf_metric.compile()
    # vmaf_neg_metric.compile()

    ### Load video 
    resolution_lq = args.TEST_INPUT_RES
    resolution_hq = args.TEST_OUTPUT_RES
    crf_ = crf if crf is not None else 22
    print(f"Testing: {video_prefix} with CRF {crf_}")

    ## Low quality video
    lq_file_path = str(test_dir_prefix) + f"/encoded{resolution_lq}CRF{crf_}/" + video_prefix + ".mp4"
    print(f"Loading {lq_file_path}")
    cap_lq = cv2.VideoCapture(lq_file_path)

    ## High quality video
    hq_file_path = str(test_dir_prefix) + f"/{video_prefix}" + ".y4m"
    cap_hq = cv2.VideoCapture(hq_file_path)

    ## Queues
    lq_queue = Queue(1) 
    hq_queue = Queue(1)

    ## Frame count
    total_frames = int(cap_hq.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap_hq.get(cv2.CAP_PROP_FPS))

    from_frame = fps * from_second
    to_frame = fps * to_second

    if to_frame is None:
        to_frame = total_frames

    to_frame = min(to_frame, total_frames)

    ## Read frames
    def read_pic(cap, q, from_frame_, to_frame_):
        count = 0
        print(cap.get(cv2.CAP_PROP_POS_MSEC))
        cap.set(cv2.CAP_PROP_POS_MSEC, from_second)
        while cap.isOpened():
            success, cv2_im = cap.read()
            if success:
                cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                x = cv2toTorch(cv2_im)
                x_bicubic = x  # torch.clip(F.interpolate(x, scale_factor=2, mode='bicubic'), min=-1, max=1)
                q.put((x, x_bicubic))
                count += 1
                if count == (to_frame_ - from_frame_):
                    print("Releasing cap")
                    cap.release()
            else:
                cap.release()

    ## Metrics
    ssim_x = []
    lpips_x = []
    vmaf_x = []
    vmaf_neg_x = []

    ## Get video resolution
    H_x = int(cap_lq.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W_x = int(cap_lq.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Src resolution: {W_x}x{H_x}")

    H_y = int(cap_hq.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W_y = int(cap_hq.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Dest resolution: {W_y}x{H_y}")

    ## Pad video
    border = 0
    modH, modW = H_x % (16 + border), W_x % (16 + border)
    padW = ((16 + border) - modW) % (16 + border)
    padH = ((16 + border) - modH) % (16 + border)

    ## Start threads
    thread1 = Thread(target=read_pic, args=(cap_lq, lq_queue, from_frame, to_frame))  # .start()
    thread2 = Thread(target=read_pic, args=(cap_hq, hq_queue, from_frame, to_frame))  # .start()
    thread1.start()
    thread2.start()

    ## Start testing
    for i in tqdm(range(to_frame - from_frame), total=to_frame - from_frame, desc=f"Testing {video_prefix}"):
        with torch.no_grad():
            # Get frames
            y_true, _ = hq_queue.get()
            x, x_bicubic = lq_queue.get()

            x = x.to(device)
            x_bicubic = x_bicubic.to(device)

            x = F.pad(x, [0, padW, 0, padH])

            x = x[:, :, :H_x, :W_x]
            x_rescaled = F.interpolate(x, scale_factor=1.5, mode='bicubic')

            ssim_loss_x = ssim(x_rescaled, y_true).mean()
            lpips_loss_x = lpips_metric(x_rescaled, y_true).mean()
            vmaf_loss_x = vmaf_metric(y_true, x_rescaled)
            vmaf_neg_loss_x = vmaf_neg_metric(y_true, x_rescaled)

            ssim_x += [float(ssim_loss_x)]
            lpips_x += [float(lpips_loss_x)]
            vmaf_x += [float(vmaf_loss_x)]
            vmaf_neg_x += [float(vmaf_neg_loss_x)]

    ### Print results
    out_dict = {
        'vid': vid, 
        'encode_res': resolution_lq, 
        'dest_res': resolution_hq
    }

    print("Mean ssim_encoded:", np.mean(ssim_x))
    print("Mean lpips_encoded:", np.mean(lpips_x))
    print("Mean vmaf_encoded:", np.mean(vmaf_x))
    print("Mean vmaf_neg_encoded:", np.mean(vmaf_neg_x))

    out_dict['ssim_encoded'] = np.mean(ssim_x)
    out_dict['lpips_encoded'] = np.mean(lpips_x)
    out_dict['vmaf_encoded'] = np.mean(vmaf_x)
    out_dict['vmaf_neg_encoded'] = np.mean(vmaf_neg_x)

    print("Evaluation complete")
    return out_dict


if __name__ == '__main__':
    test_dir = Path(args.TEST_DIR)
    videos = [vid.strip(".y4m") for vid in os.listdir(test_dir) if vid.endswith('.y4m') and '1080' in vid]

    second_start = 0
    second_finish = 120  # test no more than the 2nd minutes - none of the test videos last so much

    for crf in args.CRF:
        print(f"Testing CRF {crf}")
        output = []

        for i, vid in enumerate(tqdm(videos, desc=f"Testing Videos")):
            dict = evaluate_video(
                str(test_dir), video_prefix=vid, from_second=second_start,
                to_second=second_finish, crf=crf
            )
            output += [dict]

        df = pd.DataFrame(output)
        print(df)
        name = f"_{output[0]['encode_res']}_{output[0]['dest_res']}_TEST_CRF{crf}.csv"
        df.to_csv(name)
