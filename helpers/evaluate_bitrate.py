import pandas as pd
import utils
import os
import cv2
from pathlib import Path
from tqdm import tqdm

args = utils.ARArgs()

def evaluate_model(test_dir_prefix, video_prefix, crf=None):
    
    # Resolution of low-quality video (if needed, but we don't use it in this version)
    resolution_lq = args.TEST_INPUT_RES
    
    # Low quality video
    lq_file_path = str(test_dir_prefix) + f"/encoded{resolution_lq}CRF{crf}/" + video_prefix + ".mp4"
    cap_lq = cv2.VideoCapture(lq_file_path)
    
    # Get bitrate of the video
    video_bitrate = cap_lq.get(cv2.CAP_PROP_BITRATE)  # Get the bitrate of the video
    
    # Print and return bitrate
    print(f"Video: {video_prefix}, Bitrate: {video_bitrate} bps")
    
    # Return relevant output data
    return {'vid': video_prefix, 'bitrate': video_bitrate}


if __name__ == '__main__':
    test_dir = Path(args.TEST_DIR)
    videos = [vid.strip(".y4m") for vid in os.listdir(test_dir) if vid.endswith('.y4m') and '1080' in vid]

    crf = args.CRF  # You may adjust this as per your needs
    
    output = []
    
    for vid in tqdm(videos, desc="Testing Videos"):
        result = evaluate_model(str(test_dir), video_prefix=vid, crf=crf)
        output.append(result)

    # Convert output to DataFrame
    df = pd.DataFrame(output)
    print(df)
    
    # Save results to CSV
    name = f"bitrate_test_results_CRF{crf}.csv"
    df.to_csv(name, index=False)
    print(f"Results saved to {name}")
