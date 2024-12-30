import os
import pandas as pd
import argparse

def process_csv_files_in_directory(directory_path, output_path, all_values=False):
    try:
        # List to collect DataFrames
        all_dataframes = []

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

                    if all_values:
                        # Add a new row with the means
                        means_row = pd.DataFrame([column_means], columns=column_means.index)
                        means_row.index = ["Mean"]  # Label for the row
                        df_with_means = pd.concat([df, means_row], ignore_index=False)

                        # Add the file and directory name as columns
                        df_with_means["Source Directory"] = os.path.basename(root)
                        df_with_means["Source File"] = file

                        # Add an empty row to separate DataFrames
                        all_dataframes.append(df_with_means)
                        all_dataframes.append(pd.DataFrame())  # Empty rows between tables

                    else:
                        # Create a DataFrame with the means and file information
                        mean_df = pd.DataFrame([column_means], columns=column_means.index)
                        mean_df["Source Directory"] = os.path.basename(root)
                        mean_df["Source File"] = file

                        # Add to the total
                        all_dataframes.append(mean_df)

        # Combine all DataFrames into one
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Generate the output paths for Excel and CSV
        output_excel_path = output_path + ".xlsx"
        output_csv_path = output_path + ".csv"

        # Save the result to an Excel file
        combined_df.to_excel(output_excel_path, index=False)
        print(f"The Excel file with values {output_excel_path} was created successfully.")

        # Save the result to a CSV file
        combined_df.to_csv(output_csv_path, index=False)
        print(f"The CSV file with values {output_csv_path} was created successfully.")

    except Exception as e:
        print(f"Error during file processing: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process all CSV files in a directory and create a single file with the results.")
    parser.add_argument("--directory_path", type=str, help="Path to the directory containing the CSV files.")
    parser.add_argument("--output_path", type=str, help="Path to the output files.")
    parser.add_argument("--all_values", action="store_true", help="Include all values in the output.")
    args = parser.parse_args()

    process_csv_files_in_directory(args.directory_path, args.output_path, args.all_values)
