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
from YesNoExperimentHelper import YesNoExperimentHelper

load_dotenv(override=True)  # take environment variables from .env.


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


class ExperimentYesNo:

    def __init__(self, offline=False):
        if offline:
            self.helper = YesNoExperimentHelper(None)
            return
        self.clientExperiment = ModelClient(
            logprobs=2,
            max_tokens=3,
            modifier=MODIFIERS["yesno"],
            openAI=True,
            vertex=True,
            mistral=True,
            azure=True,
            anthropic=True,
        )
        self.clientHelper = ModelClient(openAI=True, max_tokens=4000)
        self.helper = YesNoExperimentHelper(self.clientHelper)
        # self.modelsLP = ["text-bison@002", "gpt-3.5-turbo-0125", "gpt-4"]
        # self.modelsRaw = [
        #     "claude-instant-1.2",
        #     "claude-2.1",
        #     "mistral-tiny",
        #     "mistral-medium",
        #     "llama-2-7b-chat",
        #     "llama-2-70b-chat",
        # ]
        self.modelsLP = ["gpt-4"]
        self.modelsRaw = []
        self.rawN = 10
        self.console = Console()

    def run_prompts(self, prompts, topic, run=0, chunk_size=1):
        results = []
        N = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for prompt_list in chunker(prompts, chunk_size):
                N += chunk_size

                resultsLP = []
                future_to_metadataLP = {
                    executor.submit(
                        self.clientExperiment.call_model,
                        prompt,
                        model,
                    ): (
                        prompt,
                        model,
                        topic,
                    )
                    for model in self.modelsLP
                    for prompt in prompt_list
                }
                start_time = time.time()
                for future in concurrent.futures.as_completed(future_to_metadataLP):
                    (prompt, model, topic) = future_to_metadataLP[future]
                    try:
                        response = future.result()
                        self.console.print(
                            f"[bold green] {model} generated a response:[/bold green] {response}"
                        )
                        resultsLP.append([prompt, model, topic, response])
                    except Exception as exc:
                        self.console.print(
                            f"\n[bold red] {model} generated an exception: {exc}[/bold red]"
                        )
                response_time = round(time.time() - start_time, 2)

                self.console.print(
                    f"\n[bold magenta]{run}.{N}. Fetched {len(resultsLP)} LP responses in {response_time} s:[/bold magenta] {prompt_list}."
                )

                resultsRaw = {model: [] for model in self.modelsRaw}
                future_to_metadataRaw = {
                    executor.submit(
                        self.clientExperiment.call_model,
                        prompt,
                        model,
                    ): model
                    for model in self.modelsRaw
                    for prompt in prompt_list
                    for _ in range(self.rawN)
                }
                start_time = time.time()
                for future in concurrent.futures.as_completed(future_to_metadataRaw):
                    model = future_to_metadataRaw[future]
                    try:
                        response = future.result()
                        resultsRaw[model].append(response.split()[0])
                    except Exception as exc:
                        self.console.print(
                            f"\n[bold red] {model} generated an exception: {exc}[/bold red]"
                        )
                response_time = round(time.time() - start_time, 2)

                self.console.print(
                    f"\n[bold magenta]{run}.{N}. Fetched {len(resultsRaw) * self.rawN} raw responses in {response_time} s:[/bold magenta] {prompt_list}."
                )
                resultsRawProbs = [
                    [
                        prompt,
                        model,
                        topic,
                        {
                            k: v / self.rawN
                            for k, v in dict(Counter(resultsRaw[model])).items()
                        },
                    ]
                    for model in self.modelsRaw
                    for prompt in prompt_list
                ]
                self.console.print(
                    f"[bold green] Raw Probs:[/bold green] {resultsRawProbs}"
                )
                results += resultsLP + resultsRawProbs

        return results

    def run(self):
        runs = 80
        for run in range(runs):
            topic = YESNO_TOPICS_1[random.randint(0, len(YESNO_TOPICS_1) - 1)]
            existing_prompts = None
            if (
                os.path.exists(YESNO_RESPONSES_CSV)
                and os.path.getsize(YESNO_RESPONSES_CSV) > 0
            ):
                # Get prompts for given topic if there are any
                existing_prompts = pd.read_csv(YESNO_RESPONSES_CSV)
                existing_prompts = existing_prompts[existing_prompts["topic"] == topic][
                    "prompt"
                ].tolist()

            (prompts, topic) = self.helper.generate_prompts(
                topic=topic, existing_prompts=existing_prompts
            )
            results = self.run_prompts(prompts, topic, run=run)
            self.helper.produce_dfs_from_responses(results)

    def backfill_prompts(self):
        # Backfill prompts for a given topic
        current_responses_df = pd.read_csv(YESNO_RESPONSES_CSV)
        for topic in current_responses_df["topic"].unique():
            prompts = (
                current_responses_df[current_responses_df["topic"] == topic]["prompt"]
                .unique()
                .tolist()
            )
            results = self.run_prompts(prompts, topic, chunk_size=5)
            self.helper.produce_dfs_from_responses(results)

    def clean_and_rerun_analysis(self):
        # Clean analysis DF
        analysis_df = pd.read_csv(YESNO_ANALYSIS_CSV)
        analysis_df.sort_values(
            by=["response_yes_std"], ascending=[False], inplace=True
        )
        analysis_df["prompt"] = analysis_df["prompt"].str.replace(
            r"^\d+\.\s*|-", "", regex=True
        )

        # Rerun the analysis for STD > 0.4
        prompts = analysis_df[analysis_df["response_yes_std"] > 0.4]["prompt"].tolist()
        results = self.run_prompts(prompts)
        results_enriched = self.helper.enrich_results(results)
        self.helper.produce_analysis_df(results_enriched, YESNO_CLEANED_ANALYSIS_CSV)

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
