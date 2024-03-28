import re
import ast
from typing import Dict, List
from helpers import *

from rich.console import Console
from rich.prompt import Prompt
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from ModelClient import ModelClient


load_dotenv(override=True)  # take environment variables from .env.


class PewExperimentHelper:
    def __init__(self, client: ModelClient):
        self.client = client
        self.console = Console()

    def append_and_save_df(
        self,
        df,
        file_path,
        sort_by=None,
        ascending=True,
        index=False,
        mode="a",
        header=True,
        duplicateSubset=None,
    ):
        if mode == "a" and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            existing_df = pd.read_csv(file_path)
            df = pd.concat(
                [existing_df.reset_index(drop=True), df.reset_index(drop=True)],
                ignore_index=True,
            )

        if sort_by is not None:
            df.sort_values(by=sort_by, ascending=ascending, inplace=True)

        df.drop_duplicates(inplace=True, subset=duplicateSubset)
        df.to_csv(file_path, index=index)

    def letter_from_label(self, label):
        # Map 1.0 to A, 2.0 to B, etc.
        return chr(int(label) + 64)

    def label_from_letter(self, letter):
        # Map A to 1.0, B to 2.0, etc.
        try:
            return float(format(ord(letter) - 64, ".1f"))
        except Exception as e:
            self.console.print(
                f"[bold red]Error in label_from_letter with letter {letter}: {e}[/bold red]"
            )

    def is_valid_prompt(self, prompt, debug=False):
        if not prompt:
            return False
        prompt_lower = prompt.lower()
        you_count = (
            prompt_lower.count(" you ")
            + prompt.count(" you?")
            + prompt_lower.count(" you,")
        )
        your_count = prompt_lower.count(" your ")
        you_ok_count = (
            prompt_lower.count("you think")
            + prompt_lower.count("you feel")
            + prompt_lower.count("you believe")
            + prompt_lower.count("you may know")
        )
        your_ok_count = prompt_lower.count("your opinion")

        you_ok = you_ok_count == you_count
        your_ok = your_ok_count == your_count
        if debug:
            if you_ok_count > 0 and you_count > 0:
                if you_ok:
                    self.console.print("[bold green]Prompt OK:[/bold green] " + prompt)
                else:
                    self.console.print(
                        "[bold red]Prompt rejected:[/bold red] " + prompt
                    )
            if your_ok_count > 0 and your_count > 0:
                print(your_ok_count, your_count, your_ok)
                if your_ok:
                    self.console.print("[bold green]Prompt OK:[/bold green] " + prompt)
                else:
                    self.console.print(
                        "[bold red]Prompt rejected:[/bold red] " + prompt
                    )

        if you_count == 0 and your_count == 0:
            return True
        if you_ok and your_ok:
            return True
        return False

    def build_prompt(self, question, responses):
        prompt = f"Question: {question}"
        # add each response as multiple choice option lettered with A, B, C, etc.
        for label, value in responses.items():
            if value != "Refused":
                prompt += f"\n{self.letter_from_label(label)}. {value}"
        prompt += "\nAnswer: "
        return prompt

    def valid_tokens_from_responses(self, responses):
        if not responses:
            return []
        return [
            self.letter_from_label(label)
            for label in responses.keys()
            if responses[label] != "Refused"
        ]

    def clean_special_chars(self, results_df):
        # data is a dictionary mapping strings to probabilities
        results_df["data"] = results_df["data"].apply(
            lambda x: {
                re.sub(r"[^a-zA-Z0-9\s]", "", k): v
                for k, v in x.items()
                if re.sub(r"[^a-zA-Z0-9\s]", "", k) != ""
            }
        )
        return results_df

    def validate_token_portions(self, results_df):
        results_df["valid_tokens"] = results_df["responses"].apply(
            self.valid_tokens_from_responses
        )
        results_df["valid_tokens_portion"] = results_df.apply(
            lambda x: sum(x["data"].get(token, 0) for token in x["valid_tokens"]),
            axis=1,
        )
        results_df.sort_values(by="valid_tokens_portion", ascending=False, inplace=True)
        return results_df

    def clean_dfs(self, overwrite=False):
        df = pd.read_csv(PEW_MODEL_RESULTS_CSV)
        df.drop_duplicates(inplace=True, subset=["variable", "question"])
        df.to_csv("./pew/csvs/results4_cleaned.csv", index=False)

        if overwrite:
            df.to_csv(PEW_MODEL_RESULTS_CSV, index=False)

    def cleaned_model_result(self, model_result_row):
        # First sum up the valid results
        data_l = {}
        refused_label = None
        for label, val in model_result_row["responses"].items():
            if val != "Refused":
                prob = model_result_row["data"].get(self.letter_from_label(label))
                if prob:
                    data_l[label] = prob
            else:
                refused_label = label

        # Fill with refusals if refused exists
        if refused_label:
            data_l[refused_label] = 1 - sum(data_l.values())
        else:
            # If there are no refusals, just normalize the probabilities
            data_l = {k: v / sum(data_l.values()) for k, v in data_l.items()}
        return data_l

    def clean_model_results(self):
        model_results_df = pd.read_csv(
            PEW_MODEL_RESULTS_CSV,
            converters={
                "data": lambda x: ast.literal_eval(x) if x else {},
                "responses": lambda x: ast.literal_eval(x) if x else {},
            },
        )
        # Convert the keys of data column back to labels from letters
        model_results_df["data_l"] = model_results_df.apply(
            self.cleaned_model_result, axis=1
        )
        del model_results_df["valid_tokens"]
        del model_results_df["valid_tokens_portion"]
        # Save CSV
        model_results_df.to_csv(PEW_MODEL_RESULTS_CLEANED_CSV, index=False)

    def clean_human_results(self):
        human_results_df = pd.read_csv("./pew/csvs/human_results.csv")
        human_results_df.drop_duplicates(inplace=True, subset=["variable", "question"])
        human_results_df.to_csv("./pew/csvs/human_results_cleaned.csv", index=False)

    def safe_eval_obj(self, obj):
        if not obj:
            return {}
        try:
            return ast.literal_eval(obj)
        except Exception as e:
            return {}

    def summarize_results_survey_model(self):
        compared_results_df = pd.read_csv(PEW_COMPARED_RESULTS_VALID_CSV)
        # Summarize count, average distance, STD distance per survey and model
        agg_df = (
            compared_results_df.groupby(["survey", "model"])
            .agg(
                count=pd.NamedAgg(column="variable", aggfunc="count"),
                avg_distance=pd.NamedAgg(column="distance", aggfunc="mean"),
                std_distance=pd.NamedAgg(column="distance", aggfunc="std"),
            )
            .reset_index()
        )
        # Export aggregate DF
        agg_df.to_csv(PEW_SUMMARY_RESULTS_SURVEY_MODEL, index=False)
