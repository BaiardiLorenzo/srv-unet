import argparse
from pathlib import Path

BVI_DVC_PATH = "/andromeda/datasets/BVI_DVC"
CHECKPOINTS_PATH = "/mnt/data4tb/lbaiardi/srunet_hdd/checkpoints" #"checkpoints"
TEST_DIR_PATH = "/mnt/data4tb/lbaiardi/srunet_hdd/clips"


class ARArgs:

    def __init__(self, args=None):
        ap = argparse.ArgumentParser()
        archs = ['srunet', 'unet', 'espcn', 'srresnet']
        ap.add_argument("-ds", "--dataset", type=str, default=BVI_DVC_PATH,
                        help="Dir from where to import the datasets")
        
        ap.add_argument("-m", "--model", type=str, default=None,
                        help="path to *specific* model checkpoint to load")
        
        ap.add_argument("-dv", "--device", type=str, default="0",
                        help="CUDA device to be used. For info type '$ nvidia-smi'")
        
        ap.add_argument("-v", "--verbose", action='store_true', default=False,
                        help="Verbosity mode.")
        
        ap.add_argument("--vidpatches", type=str, default="",
                        help="Where to store/load video patches.")
        
        ap.add_argument("--export", type=str, default=CHECKPOINTS_PATH,
                        help="Where to export models.")
        
        ap.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Batch size.")
        
        ap.add_argument("-e", "--epochs", type=int, default=5,
                        help="Number of epochs you want to train the model.")
        
        ap.add_argument("--arch", type=str, default="srunet", choices=archs,
                        help="Which network architecture to train.")
        
        ap.add_argument("--w0", type=float, default=1.0,
                        help="VMAF weight/LPIPS weight")
        
        ap.add_argument("--w1", type=float, default=1.0,
                        help="SSIM Weight")
        
        ap.add_argument("--w2", type=float, default=1.0,
                        help="VMAF-NEG Weight")
        
        ap.add_argument("--w3", type=float, default=1.0,
                        help="LPIPS Weight")
        
        ap.add_argument("--l0", type=float, default=0.001,
                        help="Adversarial Component Weight")
        
        ap.add_argument("--upscale", type=int, default=2,
                        help="Default upscale factor, obbtained as resolution ratio between LQ and HQ samples")
        
        ap.add_argument("--rescale_factor", action='store_true', default=False,
                        help="The groundtruth will be resized from upscale_factor*patch_size to rescale_factor*patch_size.")
        
        ap.add_argument("--layer_mult", type=float, default=1.0, 
                        help="Layer multiplier - SR UNet only")

        ap.add_argument("--n_filters", type=int, default=64, 
                        help="Net Number of filters param - SR UNet and UNet only")
        
        ap.add_argument("--downsample", type=float, default=1.0, 
                        help="Downsample factor, SR Unet and UNet only")
        
        ap.add_argument("--crf", type=int, default=22, 
                        help="Reference compression CRF")
        
        ap.add_argument('--wb-name', type=str, default='sr-unet',
                        help="Name of the Weights and Biases project")
        
        ap.add_argument('--save-critic', dest='save-critic', action='store_true',
                        help="Save the critic model")
        
        # Test only
        ap.add_argument("--testdir", type=str, default=TEST_DIR_PATH,
                        help="[TEST ONLY] Where the test clips are contained.")
        
        ap.add_argument("--testinputres", type=int, default=540, 
                        help="[TEST ONLY] Input testing resolution")
        
        ap.add_argument("--testoutputres", type=int, default=1080, 
                        help="[TEST ONLY] Output testing resolution")
        
        # Render only
        ap.add_argument('--show-only-upscaled', dest='show-only-upscaled', action='store_true',
                        help="[RENDER.PY ONLY] If you want to show only the neural net upscaled version of the video")

        ap.add_argument("--clipname", type=str, default="",
                        help="[RENDER.PY ONLY] path to the clip you want to upscale")
        
        if args is None:
            args = vars(ap.parse_args())
        else:
            args = vars(ap.parse_args(args))

        self.MODEL_NAME = args['model']
        self.VERBOSE = args['verbose']
        self.DATASET_DIR = Path(args['dataset'])
        self.CUDA_DEVICE = args['device']
        self.VID_PATCHES = args['vidpatches']
        self.EXPORT_DIR = args['export']
        self.N_EPOCHS = int(args['epochs'])
        self.BATCH_SIZE = args['batch_size']
        self.ARCHITECTURE = args['arch']
        self.VALIDATION_FREQ = 1
        self.CRF = args['crf']
        self.W0 = args['w0']
        self.W1 = args['w1']
        self.W2 = args['w2']
        self.W3 = args['w3']
        self.L0 = args['l0']
        
        self.WB_NAME = args['wb_name']
        self.SAVE_CRITIC = args['save-critic']

        self.RESCALE_FACTOR = args['rescale_factor']
        self.UPSCALE_FACTOR = args['upscale']
        self.LAYER_MULTIPLIER = args['layer_mult']
        self.N_FILTERS = args['n_filters']
        self.DOWNSAMPLE = args['downsample']

        self.TEST_INPUT_RES = args['testinputres']
        self.TEST_OUTPUT_RES = args['testoutputres']
        self.TEST_DIR = args['testdir']

        self.SHOW_ONLY_HQ = args['show-only-upscaled']
        self.CLIPNAME = args['clipname']

        self.archs = archs
        

def seed_everything(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    