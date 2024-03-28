import re
from collections import Counter
import ast
import concurrent.futures
import time
from helpers import *

from rich.console import Console
from rich.prompt import Prompt
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from ModelClient import ModelClient
from PewExperimentHelper import PewExperimentHelper
import pyreadstat
import re
from helpers import *
from scipy.stats import wasserstein_distance


load_dotenv(override=True)  # take environment variables from .env.


class ExperimentPew:
    def __init__(self):
        self.start_n = 0
        self.end_n = 120
        self.modifier = """Please answer the following question and only output the letter that corresponds to your response.\n{!r}"""

        self.client = ModelClient(
            logprobs=5,
            max_tokens=3,
            modifier=self.modifier,
            openAI=True,
            vertex=True,
            azure=True,
            anthropic=True,
            mistral=True,
        )
        self.console = Console()
        self.helper = PewExperimentHelper(self.client)
        self.modelsLP = ["text-bison@002", "gpt-3.5-turbo-0125", "gpt-4"]
        self.modelsRaw = [
            "claude-instant-1.2",
            "claude-2.1",
            "mistral-tiny",
            "mistral-medium",
            "llama-2-7b-chat",
            "llama-2-70b-chat",
        ]
        self.rawN = 10
        self.summary_df = pd.DataFrame(columns=["Survey", "ValidQ", "TotalQ"])
        self.all_valid_qs_df = pd.DataFrame()

    def import_data(self):
        # Get all SAV files in nested folders within ./pew
        if os.path.exists(PEW_MODEL_RESULTS_CSV):
            existing_surveys = pd.read_csv(PEW_MODEL_RESULTS_CSV).survey.unique()
        for dirpath, _, filenames in sorted(
            os.walk("./pew"),
            reverse=True,
        ):
            for filename in reversed(filenames):
                if filename.endswith(".sav"):
                    survey = dirpath.split("/")[-1]
                    file_n = float(survey.split("_")[0][1:])
                    if (
                        file_n < self.start_n
                        or file_n > self.end_n
                        or survey in existing_surveys
                    ):
                        continue
                    filepath = os.path.join(dirpath, filename)
                    df, meta = pyreadstat.read_sav(filepath)
                    # TODO: Handle vignette questions

                    # Export to CSV in ./pew/csvs
                    df.to_csv(f"./pew/csvs/{survey}.csv", index=False)
                    self.process_survey(survey, (df, meta), to_csv=True)

    def process_questions_df(
        self, column_names_to_labels, variable_value_labels, survey
    ):
        questions_df = pd.DataFrame(
            column_names_to_labels.items(), columns=["Variable", "Question"]
        )

        questions_df["Question"] = questions_df["Question"].apply(
            lambda x: re.sub(r"^[^.]*\.", "", x).strip() if x else x
        )
        questions_df["Demographic"] = questions_df["Question"].apply(
            lambda x: (not ("?" in x or "..." in x or "…" in x)) if x else False
        )
        questions_df["ValidQ"] = questions_df["Question"].apply(
            self.helper.is_valid_prompt
        )
        # Demographic questions are invalid
        questions_df.loc[questions_df["Demographic"], "ValidQ"] = False

        questions_df["Responses"] = questions_df["Variable"].map(
            variable_value_labels.get
        )
        questions_df["Survey"] = survey

        # Print number of valid questions and print them all
        self.console.print(
            f"Valid questions for {survey}: {questions_df['ValidQ'].sum()}"
        )
        # self.console.print(questions_df[questions_df["ValidQ"]]["Question"].values)
        return questions_df

    def process_survey(self, survey, survey_data, to_csv=True):
        # Print newlines
        self.console.print(f"Validating {survey} data")
        df, meta = survey_data
        try:
            questions_df = self.process_questions_df(
                meta.column_names_to_labels, meta.variable_value_labels, survey
            )
        except Exception as e:
            self.console.print(f"Error processing {survey}: {e}")
            return
        self.all_valid_qs_df = pd.concat(
            [
                self.all_valid_qs_df,
                questions_df[questions_df["ValidQ"]],
            ],
            ignore_index=True,
        )
        self.summary_df = pd.concat(
            [
                self.summary_df,
                pd.DataFrame(
                    {
                        "Survey": [survey],
                        "ValidQ": [questions_df["ValidQ"].sum()],
                        "TotalQ": [len(questions_df)],
                    }
                ),
            ],
            ignore_index=True,
        )
        # if to_csv:
        #     questions_df.to_csv(f"./pew/csvs/{survey}_questions.csv", index=False)
        #     self.summary_df.to_csv("./pew/csvs/summary.csv", index=False)
        #     self.all_valid_qs_df.to_csv("./pew/csvs/valid_qs.csv", index=False)

        self.run_prompts_from_df(questions_df[questions_df["ValidQ"]])

    def format_and_run_prompt(self, question, responses, model):
        # Format prompt
        prompt = self.helper.build_prompt(question, responses)
        # Run prompt
        response = self.client.call_model(prompt, model)
        return response

    def run_prompts_from_df(self, valid_qs_df):
        resultsLP = []
        resultsRawAll = []
        N = 0
        self.console.print(
            f"Running prompts for {len(valid_qs_df)} questions, {len(self.modelsLP)} LP models, and {len(self.modelsRaw) * self.rawN} raw models."
        )
        if os.path.exists(PEW_MODEL_RESULTS_CSV):
            existing_questions = pd.read_csv(PEW_MODEL_RESULTS_CSV).question.unique()
            valid_qs_df = valid_qs_df[~valid_qs_df["Question"].isin(existing_questions)]

        if valid_qs_df.empty:
            self.console.print(
                "[bold red]No new questions to run prompts for[/bold red]"
            )
            return

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for row in valid_qs_df.itertuples(index=False):
                N += 1
                future_to_metadataLP = {
                    executor.submit(
                        self.format_and_run_prompt,
                        row.Question,
                        row.Responses,
                        model,
                    ): model
                    for model in self.modelsLP
                }

                start_time = time.time()
                for future in concurrent.futures.as_completed(future_to_metadataLP):
                    model = future_to_metadataLP[future]
                    try:
                        data = future.result()
                        resultsLP.append(
                            [
                                row.Survey,
                                row.Variable,
                                row.Question,
                                row.Responses,
                                model,
                                data,
                            ]
                        )
                    except Exception as exc:
                        self.console.print(
                            f"\n[bold red] We generated an exception: {exc}[/bold red]"
                        )
                response_time = round(time.time() - start_time, 2)
                self.console.print(
                    f"\n[bold magenta]{N}. Fetched {len(self.modelsLP)} LP responses in {response_time} s:[/bold magenta] {row.Question}."
                )

                resultsRaw = {model: [] for model in self.modelsRaw}
                future_to_metadataRaw = {
                    executor.submit(
                        self.format_and_run_prompt,
                        row.Question,
                        row.Responses,
                        model,
                    ): model
                    for model in self.modelsRaw
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
                    f"\n[bold magenta]{N}. Fetched {len(self.modelsRaw) * self.rawN} raw responses in {response_time} s:[/bold magenta] {row.Question}."
                )

                resultsRawProbs = [
                    [
                        row.Survey,
                        row.Variable,
                        row.Question,
                        row.Responses,
                        model,
                        {
                            k: v / self.rawN
                            for k, v in dict(Counter(resultsRaw[model])).items()
                        },
                    ]
                    for model in self.modelsRaw
                ]
                self.console.print(f"[bold green] Raw Probs:[/bold green]")
                print([x[4:6] for x in resultsRawProbs])
                resultsRawAll += resultsRawProbs
        results = resultsLP + resultsRawAll
        if len(results) == 0:
            self.console.print("[bold red]No new results to add to the CSV[/bold red]")
            return
        results_df = pd.DataFrame(
            results,
            columns=[
                "survey",
                "variable",
                "question",
                "responses",
                "model",
                "data",
            ],
        )
        results_df = self.helper.clean_special_chars(results_df)
        results_df = self.helper.validate_token_portions(results_df)
        self.console.print(results_df)

        self.helper.append_and_save_df(
            results_df, PEW_MODEL_RESULTS_CSV, duplicateSubset=["variable", "model"]
        )

    def run_all_prompts(self):
        return self.run_prompts_from_df(self.all_valid_qs_df)

    def process_human_data_df(self, questions_df, human_data_df):
        # Get column that starts with WEIGHT_
        weights_col = human_data_df.columns[
            human_data_df.columns.str.startswith("WEIGHT_")
        ][0]
        weights = human_data_df[weights_col]

        validq_variables = questions_df[questions_df["ValidQ"]]["Variable"]
        all_frequencies = []
        existing_variables = (
            pd.read_csv("./pew/csvs/human_results.csv").variable.unique()
            if os.path.exists("./pew/csvs/human_results.csv")
            else []
        )

        for variable in validq_variables:
            if variable in existing_variables:
                continue
            frequencies = {}
            value_options = (
                questions_df.loc[questions_df["Variable"] == variable, "Responses"]
                .iloc[0]
                .keys()
            )
            for value in value_options:
                # Calculate the weighted frequency for the value

                weighted_frequency = (
                    (human_data_df[variable] == value).mul(weights).sum()
                )
                frequencies[value] = weighted_frequency
            frequencies = {
                k: v / sum(frequencies.values()) for k, v in frequencies.items()
            }
            all_frequencies.append([variable, frequencies])

        human_frequency_df = pd.DataFrame(
            all_frequencies, columns=["variable", "data_l"]
        )
        human_frequency_df["model"] = "humans-all"

        # Join with questions_df to get columns survey, question, and responses
        human_frequency_df = human_frequency_df.merge(
            questions_df[["Variable", "Survey", "Question", "Responses"]],
            left_on="variable",
            right_on="Variable",
        )
        del human_frequency_df["Variable"]
        # Make columns lowercase
        human_frequency_df.columns = map(str.lower, human_frequency_df.columns)

        # Print the results
        print(human_frequency_df)
        return human_frequency_df

    def import_human_data(self):
        top_surveys = pd.read_csv("./pew/csvs/summary.csv").Survey.unique()
        for i, survey in enumerate(top_surveys):
            self.console.print(
                f"[bold green]{i} / Processing human data for {survey}[/bold green]"
            )
            questions_df = pd.read_csv(
                f"./pew/csvs/{survey}_questions.csv",
                converters={"Responses": lambda x: ast.literal_eval(x) if x else {}},
            )
            human_data_df = pd.read_csv(f"./pew/csvs/{survey}.csv")
            # Skip if DFs are empty
            if human_data_df.empty or questions_df.empty:
                continue

            human_frequency_df = self.process_human_data_df(questions_df, human_data_df)
            filepath = "./pew/csvs/human_results.csv"
            self.helper.append_and_save_df(
                human_frequency_df, filepath, duplicateSubset=["variable", "model"]
            )

    def compare_human_model_results(self):
        human_results_df = pd.read_csv(
            "./pew/csvs/human_results_cleaned.csv",
            converters={"data_l": self.helper.safe_eval_obj},
        )
        model_results_df = pd.read_csv(
            PEW_MODEL_RESULTS_CLEANED_CSV,
            converters={
                "data_l": self.helper.safe_eval_obj,
                "responses": self.helper.safe_eval_obj,
            },
        )
        compared_results = []
        # TODO: Handle non ordinal responses
        for model in model_results_df["model"].unique():
            current_model_results_df = model_results_df[
                model_results_df["model"] == model
            ].copy()
            # Print DF length
            self.console.print(
                f"Comparing {len(human_results_df)} human results with {len(current_model_results_df)} model results for {model}"
            )
            combined_df = pd.merge(
                human_results_df,
                current_model_results_df,
                on="variable",
                suffixes=("_human", "_model"),
            )
            for index, row in combined_df.iterrows():
                # Handle refused labels
                refused_label = None
                for label, val in row["responses_model"].items():
                    if val == "Refused":
                        refused_label = label
                        break

                data_human_values = [
                    x for x in row["data_l_human"].keys() if x != refused_label
                ]
                data_model_values = [
                    x for x in row["data_l_model"].keys() if x != refused_label
                ]
                data_human_weights = [row["data_l_human"][x] for x in data_human_values]
                data_model_weights = [row["data_l_model"][x] for x in data_model_values]

                if len(data_human_values) == 0 or len(data_model_values) == 0:
                    continue

                distance = wasserstein_distance(
                    data_human_values,
                    data_model_values,
                    u_weights=data_human_weights,
                    v_weights=data_model_weights,
                )

                # Generate summary
                top_human = max(row["data_l_human"].items(), key=lambda x: x[1])
                top_human_response = row["responses_model"][top_human[0]]
                top_human_prob = "{:.2%}".format(top_human[1])

                top_model = max(row["data_l_model"].items(), key=lambda x: x[1])
                top_model_response = row["responses_model"][top_model[0]]
                top_model_prob = "{:.2%}".format(top_model[1])

                summary = f"""Human: {top_human_response} @ {top_human_prob}, Model: {top_model_response} @ {top_model_prob}, Distance: {distance}"""

                validq2 = (
                    " mention" not in row["question_human"].lower()
                    and "you…" not in row["question_human"].lower()
                    and "coded" not in row["question_human"].lower()
                    and "THERMO_" not in row["variable"]
                )
                opinion_or_fact = (
                    "opinion"
                    if (
                        "you " in row["question_human"].lower()
                        or "your " in row["question_human"].lower()
                        or "you." in row["question_human"].lower()
                        or "you," in row["question_human"].lower()
                        or "you…" in row["question_human"].lower()
                    )
                    else "fact"
                )
                model_responded = (
                    top_model_response.lower() != "refused"
                    and top_model_response.lower() != "not sure"
                )

                compared_results.append(
                    {
                        "variable": row["variable"],
                        "question": row["question_human"],
                        "summary": summary,
                        "model": row["model_model"],
                        "responses": row["responses_model"],
                        "distance": distance,
                        "human_data_l": row["data_l_human"],
                        "model_data_l": row["data_l_model"],
                        "survey": row["survey_human"],
                        "validq2": validq2,
                        "opinion_or_fact": opinion_or_fact,
                        "model_responded": model_responded,
                    }
                )

        compared_results_df = pd.DataFrame(compared_results)
        compared_results_df.sort_values(by="distance", ascending=False, inplace=True)
        compared_results_df.to_csv(PEW_COMPARED_RESULTS_CSV, index=False)

        compared_results_valid_df = compared_results_df[
            compared_results_df["validq2"]
            & (compared_results_df["opinion_or_fact"] == "opinion")
            & compared_results_df["model_responded"]
        ]
        compared_results_valid_df.to_csv(PEW_COMPARED_RESULTS_VALID_CSV, index=False)

    def clean(self):
        self.helper.clean_dfs(overwrite=True)

    def clean_human_results(self):
        self.helper.clean_human_results()

    def clean_model_results(self):
        self.helper.clean_model_results()

    def run(self):
        # self.import_data()
        # self.import_human_data()
        # self.clean_human_results()
        # self.clean_model_results()
        # self.clean()
        self.compare_human_model_results()
        # self.helper.summarize_compared_results()
