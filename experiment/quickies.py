import pandas as pd


def process_csvs_pew_v12(input_filepath, output_filepath):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(
        input_filepath, sep="\t"
    )  # Adjust sep according to your actual delimiter

    # Define a function to extract and format the desired information from the 'summary' column
    def extract_and_format_summary(summary):
        parts_0 = summary.split(",")
        parts = [parts_0[0], ",".join(parts_0[1:-1]), parts_0[-1]]
        human_summary = parts[0].split("@")[1].strip()
        model_summary = parts[1].split("@")[1].strip()
        print(human_summary, model_summary)
        distance = parts[2].split(":")[1].strip()
        distance = round(float(distance), 2)
        # Extract percentages and round them
        human_percent = round(float(human_summary.split("%")[0]))
        model_percent = round(float(model_summary.split("%")[0]))
        # Extract choices
        human_choice = parts[0].split(" @")[0].split(": ")[1]
        model_choice = parts[1].split(" @")[0].split(": ")[1]
        print(human_choice, model_choice)
        return (
            f"{human_choice} ({human_percent}%)",
            f"{model_choice} ({model_percent}%)",
            distance,
        )

    # Apply the extraction and formatting function to each row in the 'summary' column
    # And split the results into three new columns
    df[["Top Choice: Americans", "Top Choice: GPT 3.5", "Distance"]] = df.apply(
        lambda row: pd.Series(extract_and_format_summary(row["summary"])), axis=1
    )

    # Select and rename the columns as required
    output_df = df[
        ["question", "Top Choice: Americans", "Top Choice: GPT 3.5", "Distance"]
    ].rename(columns={"question": "Prompt"})

    # Save the resulting DataFrame to a new CSV file
    output_df.to_csv(output_filepath, index=False)


# Example usage:
input_filepath = "gpt3_v12_1.tsv"  # Replace with your input file path
output_filepath = "gpt3_v12_output_1.csv"  # Replace with your desired output file path
process_csvs_pew_v12(input_filepath, output_filepath)
