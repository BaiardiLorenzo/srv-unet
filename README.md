# SRV-UNet
It is an architecture comprised with a GAN-based training procedure for obtaining a fast neural network which enable better 
bitrate performances respect to the H.265 codec for the same quality, or better quality at the same bitrate.
It is different from the original SR-UNet architecture in the sense that it uses a VMAF/VMAF-NEG loss function for training.

## Requirements
- Installing requirements: `$ pip install -r requirements.txt`
- [VMAF Pytorch](https://github.com/alvitrioliks/VMAF-torch): Follow the instructions in the repository to install the VMAF
  metric for PyTorch.
- FFMpeg compiled with H.265 codec and also VMAF metric. For references
  check [the official compilation guide](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu).
- [VMAF GitHub Repository](https://github.com/Netflix/vmaf).
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity): For the original implementation of [1].

## Dataset
The dataset used for training is the [BVI-DVC](https://arxiv.org/pdf/2003.13552). 
For preparing the dataset there are two helper script, `compress_train_videos.sh` for spatially compressing and encoding each video, 
then with `extract_train_frames.sh` the dataset can be prepared. 
The train dataset should follow this naming scheme (assuming the videos are encoded with CRF 22):

```bash
  [DATASET_DIR]/
      frames_HQ/
          [clipName1]/
              [clipName1]_001.png
              [clipName1]_002.png
              ...
              [clipName1]_064.png
          [clipName2]/
              ...
          [clipNameN]/
              ...
      frames_CRF_22/
          [clipName1]/
              [clipName1]_001.png
              [clipName1]_002.png
              ...
              [clipName1]_064.png
          [clipName2]/
              ...
          [clipNameN]/
              ...
```

## Training
To train the model for 2x Super Resolution (as used in the model for the 540p -> 1080p upscaling):

```bash
$ python train.py --dataset [DATASET_DIR]
```

For performing an easier 1.5x upscale (720p -> 1080p):

```bash
$ python train.py --arch srunet --layer_multiplier 0.7 --n_filters 48 --downsample 0.75 --device 0 \
--upscale 2 --export [EXPORT_DIR] --epochs 80 --dataset [DATASET_DIR] --crf [CRF]
```

For more information, inspect `utils.py`.

## Testing
We tested on the 1080p clips available from the [Derf's Collection](https://media.xiph.org/video/derf/) in Y4M format. 
For preparing the test set (of encoded clips) you can use the `compress_test_videos.sh` helper script.
The test set will be structured as follows, and there is no need of extracting each frame:

```bash
    [TEST_DIR]/
        encoded540CRF23/
            aspen_1080p.mp4
            crowd_run_1080p50.mp4
            ducks_take_off_1080p50.mp4
            ...
            touchdown_pass_1080p.mp4
        aspen_1080p.y4m
        crowd_run_1080p50.y4m
        ducks_take_off_1080p50.y4m
        ...
        touchdown_pass_1080p.y4m
```

For testing the model:

```bash
$ python evaluate_model.py --model [MODEL_NAME] --arch srunet --layer_mult 0.7 --n_filters 48 \
--downsample 0.75 --device 0 --upscale 2 --crf 22 --testdir [TEST_DIR] --testinputres 720 --testoutputres 1080
```

## References
- [1] Fast video visual quality and resolution improvement using SR-UNet. Authors Federico Vaccaro, Marco Bertini,
  Tiberio Uricchio, and Alberto Del Bimbo.
  