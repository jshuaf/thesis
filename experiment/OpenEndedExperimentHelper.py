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


class OpenEndedExperimentHelper:
    def __init__(self):
        self.clientPrompts = ModelClient(openAI=True, max_tokens=4000)
        self.clientCoder = ModelClient(openAI=True, max_tokens=1)
        self.console = Console()

    def generate_prompts_harm(self, harm_instruction, harm_type):
        prompt = HARM_PROMPT.format(
            HARM_INSTRUCTIONS.get(harm_instruction), HARM_TYPES.get(harm_type)
        )
        model = "gpt-3.5-turbo-0125"
        prompts_raw = self.clientPrompts.call_model(prompt, model)

        # Process and clean prompts
        prompts = prompts_raw.split("\n")
        prompts = [x for x in prompts if len(x) > 0]
        prompts = [re.sub(r"^\d+\.|-", "", x) for x in prompts]
        return prompts

    def generate_prompts_advice(self, advice_option):
        prompt = ADVICE_PROMPT.format(advice_option)
        model = "gpt-3.5-turbo-0125"
        prompts_raw = self.clientPrompts.call_model(prompt, model)

        # Process and clean prompts
        prompts = prompts_raw.split("\n")
        prompts = [x for x in prompts if len(x) > 0]
        prompts = [re.sub(r"^\d+\.|-", "", x) for x in prompts]
        return prompts

    def code_response(self, prompt, response):
        code_prompt = REFUSAL_CODER_PROMPT.format(prompt, response)
        model = "gpt-3.5-turbo-0125"
        coded = self.clientCoder.call_model(code_prompt, model)

        # Process and clean response
        if coded in ["A", "B", "C", "D", "E"]:
            return REFUSAL_OPTIONS.get(coded)
        else:
            # Print in red with response and coded
            self.console.print(f"[red]{response}[/red] was coded as [red]{coded}[/red]")
            return "n/a"

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

        if duplicateSubset:
            df.drop_duplicates(inplace=True, subset=duplicateSubset, keep="last")
        df.to_csv(file_path, index=index)

    def enrich_results(self, results):
        current_dt = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
        results_enriched = [
            x + [self.get_yesp_from_probs(x[3]), current_dt] for x in results
        ]
        return results_enriched

    def produce_dfs_from_responses(self, results, file_path):
        results_df = self.produce_results_df(results, file_path)
        if file_path == OPENENDED_HARM_RESULTS_CSV:
            self.produce_analysis_df(results_df, OPENENDED_HARM_ANALYSIS_CSV)
        elif file_path == OPENENDED_ADVICE_RESULTS_CSV:
            self.produce_analysis_df(results_df, OPENENDED_ADVICE_ANALYSIS_CSV)

    def produce_results_df(self, results, file_path):
        current_dt = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
        results_df = pd.DataFrame(
            results,
            columns=[
                "prompt",
                "model",
                "topic",
                "response",
                "coded",
            ],
        )
        results_df["datetime"] = current_dt
        self.append_and_save_df(
            results_df,
            file_path,
        )
        return results_df

    def produce_analysis_df(self, results_df, path):
        # ANALYSIS DATAFRAME
        results_df["refused"] = results_df["coded"].apply(
            lambda x: 1 if "refusal" in x or "referral" in x else 0
        )
        summary_df = (
            results_df.groupby(["prompt", "model", "topic"])
            .agg(
                refused_prob=pd.NamedAgg(column="refused", aggfunc="mean"),
                count=pd.NamedAgg(column="refused", aggfunc="count"),
            )
            .reset_index()
        )

        pivot_df = summary_df.pivot_table(
            index="prompt", columns="model", values="refused_prob", aggfunc="mean"
        )
        agg_df = (
            summary_df.groupby(["prompt"])
            .agg(
                refused_prob_mean=pd.NamedAgg(column="refused_prob", aggfunc="mean"),
                refused_prob_std=pd.NamedAgg(column="refused_prob", aggfunc="std"),
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

        analysis_df.sort_values(
            by=["refused_prob_std"], ascending=[False], inplace=True
        )

        self.append_and_save_df(
            analysis_df,
            path,
            sort_by="refused_prob_std",
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
