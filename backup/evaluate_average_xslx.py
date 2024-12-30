import os
import pandas as pd
import argparse


def process_csv_files_in_directory(directory_path, output_path):
    try:
        # List to collect DataFrames
        mean_dataframes = []

        # Walk through the directory and subdirectories
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    print(f"Processing file: {file_path}")

                    # Load the CSV file
                    df = pd.read_csv(file_path)

                    # Calculate the mean of numeric columns
                    column_means = df.select_dtypes(include='number').mean()

                    # Create a DataFrame with the means and file information
                    mean_df = pd.DataFrame([column_means], columns=column_means.index)
                    mean_df["Source Directory"] = os.path.basename(root)
                    mean_df["Source File"] = file

                    # Add to the total
                    mean_dataframes.append(mean_df)

        # Combine all DataFrames into one
        combined_means_df = pd.concat(mean_dataframes, ignore_index=True)

        # Generate the output paths for Excel and CSV
        output_excel_path = output_path + ".xlsx"
        output_csv_path = output_path + ".csv"

        # Save the result to an Excel file
        combined_means_df.to_excel(output_excel_path, index=False)
        print(f"The Excel file with mean values {output_excel_path} was created successfully.")

        # Save the result to a CSV file
        combined_means_df.to_csv(output_csv_path, index=False)
        print(f"The CSV file with mean values {output_csv_path} was created successfully.")

    except Exception as e:
        print(f"Error during file processing: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process all CSV files in a directory and create a single Excel file with the mean values.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing the CSV files.")
    parser.add_argument("output_path", type=str, help="Path to the output files.")
    args = parser.parse_args()

    process_csv_files_in_directory(args.directory_path, args.output_path)
