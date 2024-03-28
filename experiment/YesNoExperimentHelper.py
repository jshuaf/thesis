import re
from typing import Dict, List
from helpers import *

from rich.console import Console
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from ModelClient import ModelClient
import random

load_dotenv(override=True)  # take environment variables from .env.

class YesNoExperimentHelper:
    def __init__(self, client: ModelClient):
        self.client = client
        self.console = Console()

    def generate_prompts(self, existing_prompts=None, topic=None):
        if not topic:
            topic = YESNO_TOPICS_1[random.randint(0, len(YESNO_TOPICS_1) - 1)]
        prompt = YESNO_PROMPT.format(topic)

        if existing_prompts:
            prompt += (
                "\nHere are some existing prompts that have prompted good disagreement, please use these as a model but avoid generating prompts that are too similar to what we already have:\n"
                + "\n".join(existing_prompts)
            )
        model = "gpt-3.5-turbo-0125"
        prompts_raw = self.client.call_model(prompt, model)

        # Process and clean prompts
        prompts = prompts_raw.split("\n")
        prompts = [x for x in prompts if len(x) > 0]
        prompts = [re.sub(r"^\d+\.|-", "", x) for x in prompts]
        return (prompts, topic)

    def get_yesp_from_probs(self, logprobs: Dict[str, float]) -> Dict[str, float]:
        # If one of the keys is like "Yes" or "No", just use that one
        for k in logprobs.keys():
            if "Yes" in k and logprobs[k] >= 0.5:
                return logprobs[k]
            if "No" in k and logprobs[k] >= 0.5:
                return 1 - logprobs[k]
        # If not, just return a probability of -1 to indicate a refusal
        return -1

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
            df = pd.concat([existing_df, df], ignore_index=True)

        if sort_by is not None:
            df.sort_values(by=sort_by, ascending=ascending, inplace=True)

        if mode == "a":
            header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0

        df.drop_duplicates(inplace=True, subset=duplicateSubset, keep="last")
        df.to_csv(file_path, index=index)

    def enrich_results(self, results):
        current_dt = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
        results_enriched = [
            x + [self.get_yesp_from_probs(x[3]), current_dt] for x in results
        ]
        return results_enriched

    def produce_dfs_from_responses(self, results):
        results_df = self.produce_results_df(results)
        self.produce_analysis_df(results_df)

    def produce_results_df(self, results):
        current_dt = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
        results_df = pd.DataFrame(
            results,
            columns=[
                "prompt",
                "model",
                "topic",
                "response",
            ],
        )
        results_df["datetime"] = current_dt
        results_df["response_yes"] = results_df["response"].apply(
            self.get_yesp_from_probs
        )
        self.append_and_save_df(
            results_df, YESNO_RESPONSES_CSV, duplicateSubset=["prompt", "model"]
        )
        return results_df

    def produce_analysis_df(self, results_df, path=None):
        # ANALYSIS DATAFRAME
        if not path:
            path = YESNO_ANALYSIS_CSV

        results_df = results_df[results_df["response_yes"] != -1]
        pivot_df = results_df.pivot_table(
            index="prompt", columns="model", values="response_yes", aggfunc="mean"
        )
        agg_df = (
            results_df.groupby(["prompt"])
            .agg(
                response_yes_mean=pd.NamedAgg(column="response_yes", aggfunc="mean"),
                response_yes_std=pd.NamedAgg(column="response_yes", aggfunc="std"),
            )
            .reset_index()
        )

        analysis_df = pd.merge(agg_df, pivot_df, on="prompt")
        analysis_df = pd.merge(
            analysis_df,
            results_df[["prompt", "topic"]].drop_duplicates(),
            on="prompt",
            how="left",
        )

        analysis_df.rename(
            columns={col: f"{col} yesp" for col in pivot_df.columns}, inplace=True
        )
        analysis_df.sort_values(
            by=["response_yes_std"], ascending=[False], inplace=True
        )

        self.append_and_save_df(
            analysis_df,
            path,
            sort_by="response_yes_std",
            ascending=False,
            duplicateSubset="prompt",
        )

    def produce_topics_df(self, results):
        # TOPICS DATAFRAME
        working_results_df = pd.read_csv(YESNO_RESPONSES_CSV)

        topics_df = (
            working_results_df.groupby("topic")
            .agg(
                avg_std=pd.NamedAgg(
                    column="response_yes", aggfunc=lambda x: np.std(x, ddof=1)
                )
            )
            .sort_values(by="avg_std", ascending=False)
            .reset_index()
        )
        self.append_and_save_df(
            topics_df,
            TOPICS_CSV,
            sort_by="avg_std",
            ascending=False,
            mode="w",
        )

    def is_valid_prompt(self, prompt, debug=False):
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
