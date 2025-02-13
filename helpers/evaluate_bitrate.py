import os
import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import utils

args = utils.ARArgs()


def calculate_bitrate(directory, resolution, crf):
    """
    Calculate the average bitrate for all video files in a specific folder of type encoded{resolution}CRF{crf}.
    
    Parameters:
        directory (str): Path of the main directory containing the folders.
        resolution (str): Resolution of the videos.
        crf (str): CRF of the videos.
    
    Returns:
        float: Average bitrate of the videos in the specified folder.
    """
    folder_name = f"encoded{resolution}CRF{crf}"
    folder_path = Path(directory) / folder_name

    if not folder_path.exists():
        print(f"The folder {folder_path} does not exist, skipped.")
        return None

    video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
    if not video_files:
        print(f"No video files found in the folder {folder_path}, skipped.")
        return None

    total_bitrate = 0
    valid_files = 0

    for video in tqdm(video_files, desc=f"Calculating bitrate for {folder_name}"):
        video_path = folder_path / video
        cap = cv2.VideoCapture(str(video_path))
        bitrate = cap.get(cv2.CAP_PROP_BITRATE)  # Bitrate in bps
        cap.release()

        if bitrate > 0:
            total_bitrate += bitrate
            valid_files += 1
        else:
            print(f"Unable to calculate bitrate for {video}")

    if valid_files > 0:
        avg_bitrate = total_bitrate / valid_files
        print(f"Average bitrate for videos in {folder_name}: {avg_bitrate:.2f} bps")
        return avg_bitrate
    else:
        print(f"No valid bitrate calculated for {folder_name}")
        return None


if __name__ == '__main__':
    directory = Path(args.TEST_DIR)

    # Lists of resolutions and CRF
    resolutions = ["540", "720", "1080"]
    crf_values = ["22", "25", "30", "33", "37", "40", "42", "45"]

    results = []

    for resolution in resolutions:
        for crf in crf_values:
            try:
                avg_bitrate = calculate_bitrate(directory, resolution=resolution, crf=crf)
                if avg_bitrate is not None:
                    results.append({
                        "resolution": resolution,
                        "crf": crf,
                        "avg_bitrate": avg_bitrate
                    })
            except Exception as e:
                print(f"Error during calculation for encoded{resolution}CRF{crf}: {e}")

    # Creating the final DataFrame
    results_df = pd.DataFrame(results)

    # Saving to a single CSV
    output_csv = "bitrate_results_summary.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved in {output_csv}")

    # Saving to a single Excel
    output_excel = "bitrate_results_summary.xlsx"
    results_df.to_excel(output_excel, index=False)
    print(f"Results saved in {output_excel}")
