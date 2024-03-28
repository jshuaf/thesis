import concurrent.futures
import time
from typing import Dict, List
from helpers import *
from collections import Counter
import random

from rich.console import Console
from rich.prompt import Prompt
import pandas as pd
import os
from dotenv import load_dotenv
from ModelClient import ModelClient
from OpenEndedExperimentHelper import OpenEndedExperimentHelper

load_dotenv(override=True)  # take environment variables from .env.


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


class ExperimentOpenEnded:

    def __init__(self, offline=False):
        if offline:
            self.helper = OpenEndedExperimentHelper(None)
            return
        self.clientExperiment = ModelClient(
            logprobs=2,
            max_tokens=100,
            modifier=MODIFIERS["openended"],
            openAI=True,
            vertex=True,
            mistral=True,
            azure=True,
            anthropic=True,
        )
        self.helper = OpenEndedExperimentHelper()
        self.modelsRaw = [
            "claude-instant-1.2",
            "claude-2.1",
            "mistral-tiny",
            "mistral-medium",
            "llama-2-7b-chat",
            "llama-2-70b-chat",
            "text-bison@002",
            "gpt-3.5-turbo-0125",
            "gpt-4",
        ]
        self.rawN = 3
        self.console = Console()

    def run_prompts(self, prompts, topic, run=0, chunk_size=1):
        results = []
        N = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for prompt_list in chunker(prompts, chunk_size):
                N += chunk_size

                resultsRaw = []
                future_to_metadataRaw = {
                    executor.submit(
                        self.clientExperiment.call_model,
                        prompt,
                        model,
                    ): (model, prompt)
                    for model in self.modelsRaw
                    for prompt in prompt_list
                    for _ in range(self.rawN)
                }
                start_time = time.time()
                for future in concurrent.futures.as_completed(future_to_metadataRaw):
                    (model, prompt) = future_to_metadataRaw[future]
                    try:
                        response = future.result()
                        resultsRaw.append([prompt, model, topic, response])
                    except Exception as exc:
                        self.console.print(
                            f"\n[bold red] {model} generated an exception: {exc}[/bold red]"
                        )
                response_time = round(time.time() - start_time, 2)

                self.console.print(
                    f"\n[bold magenta]{run}.{N}. Fetched {len(resultsRaw)} raw responses in {response_time} s:[/bold magenta] {prompt_list}."
                )
                results += resultsRaw

        return results

    def code_results(self, results, chunk_size=15):
        N = 0
        coded_results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for results_list in chunker(results, chunk_size):
                N += chunk_size
                future_to_metadata = {
                    executor.submit(
                        self.helper.code_response,
                        result[0],
                        result[3],
                    ): result
                    for result in results_list
                }
                start_time = time.time()
                for future in concurrent.futures.as_completed(future_to_metadata):
                    response = future_to_metadata[future]
                    try:
                        coded = future.result()
                        coded_results.append([*response, coded])
                    except Exception as exc:
                        self.console.print(
                            f"\n[bold red] {response} generated an exception: {exc}[/bold red]"
                        )
                response_time = round(time.time() - start_time, 2)
                self.console.print(
                    f"\n[bold magenta]{N}. Coded {len(results_list)} responses in {response_time} s.[/bold magenta]"
                )
        return coded_results

    def run_harm(self):
        for harm_instruction in sorted(HARM_INSTRUCTIONS.keys()):
            for harm_type in sorted(HARM_TYPES.keys()):
                prompts = self.helper.generate_prompts_harm(harm_instruction, harm_type)
                topic = f"{harm_instruction}-{harm_type}"
                results = self.run_prompts(prompts, topic, chunk_size=5)
                coded_results = self.code_results(results)

                self.helper.produce_dfs_from_responses(
                    coded_results, OPENENDED_HARM_RESULTS_CSV
                )
                print(harm_instruction, harm_type, prompts, sep="\n")

    def run_advice(self):
        for advice_option in ADVICE_OPTIONS:
            prompts = self.helper.generate_prompts_advice(advice_option)
            topic = advice_option
            results = self.run_prompts(prompts, topic, chunk_size=5)
            coded_results = self.code_results(results)

            self.helper.produce_dfs_from_responses(
                coded_results, OPENENDED_ADVICE_RESULTS_CSV
            )
            print(advice_option, prompts, sep="\n")

    def run(self):
        # self.run_harm()
        self.run_advice()

    def generate_model_agreement(self):
        responses_df = pd.read_csv(YESNO_RESPONSES_CSV)
        models = responses_df["model"].unique()
        analysis_df = pd.read_csv(YESNO_ANALYSIS_CSV)
        model_agreement_df = pd.DataFrame(index=models, columns=models)
        # Calculate the average of the differences for each model pair
        for model_1 in models:
            for model_2 in models:
                # Difference between each pair of models, row-wise
                differences = (
                    analysis_df[model_1 + " yesp"] - analysis_df[model_2 + " yesp"]
                )
                # Calculate the average of these differences
                avg_diff = differences.mean()
                # Populate the comparison DataFrame
                model_agreement_df.at[model_1, model_2] = abs(avg_diff)
        model_agreement_df.to_csv(YESNO_MODEL_AGREEMENT_CSV, index=True)

    def generate_model_duo_analysis_df(self, model1, model2):
        current_responses_df = pd.read_csv(YESNO_RESPONSES_CSV)
        filtered_df = current_responses_df[
            (current_responses_df["model"] == model1)
            | (current_responses_df["model"] == model2)
        ]
        path = f"./yesno/experiment_analysis_{model1}_{model2}_{YESNO_VERSION}.csv"
        self.helper.produce_analysis_df(filtered_df, path=path)
