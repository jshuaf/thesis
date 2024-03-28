import csv
import time
from helpers import *

from rich.console import Console
from rich.prompt import Prompt
import pandas as pd
from dotenv import load_dotenv
from ModelClient import ModelClient
from ExperimentOpenEnded import ExperimentOpenEnded

load_dotenv(override=True)  # take environment variables from .env.


def test_anthropic():
    print("testing")
    client = ModelClient(logprobs=False, vertex=False, anthropic=True)
    print("hi")
    res = client.call_model("Hello, Claude.", "claude-2")
    print(res)


def test_mistral():
    print("testing mistral")
    client = ModelClient(logprobs=False, vertex=False, mistral=True)
    res = client.call_model("Hello, Mistral. What's the weather today.", "mistral-tiny")
    print(res)


# def test_anthropic_new():
#     service_account_file = "./company-data-414205-d72153d6a515.json"
#     credentials = service_account.Credentials.from_service_account_file(
#         service_account_file
#     )

#     client = AnthropicVertex(region="us-central1", project_id="company-data-414205")
#     message = client.messages.create(
#         max_tokens=1024,
#         messages=[
#             {
#                 "role": "user",
#                 "content": "Send me a recipe for banana bread.",
#             }
#         ],
#         model="claude-instant-1p2",
#     )
#     print(message.model_dump_json(indent=2))


def prompter():
    client = ModelClient(logprobs=False, vertex=False, modifier=None)
    while True:
        console = Console()
        console.print(
            "[bold green]Enter your prompt (or type 'exit' to quit):[/bold green]"
        )
        prompt = Prompt.ask("Prompt")
        models = ["text-bison@002", "gpt-3.5-turbo-0125"]
        responses = []

        for model in models:
            start_time = time.time()
            response = client.call_model(prompt, model)
            response_time = round(time.time() - start_time, 2)

            responses.append(response)
            color = MODELS_TO_COLORS[model]
            console.print(
                f"\n[bold {color}] {model} Response ({response_time} s):[/bold {color}]",
                response,
            )

        with open("model_responses.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([prompt] + responses)

        console.print("\n---------------------------------------\n")


def clean_dfs():
    # Remove duplicates from all DFs
    for csv in [
        YESNO_RESPONSES_CSV,
        YESNO_ANALYSIS_CSV,
        YESNO_TOPICS_CSV,
    ]:
        df = pd.read_csv(csv)
        df.drop_duplicates(inplace=True)
        df.to_csv(csv, index=False)


def main():
    # experiment = ExperimentYesNo(offline=True)
    # experiment.generate_model_agreement()
    # experiment.backfill_prompts()
    # experiment.generate_model_duo_analysis_df("gpt-3.5-turbo-0125", "gpt-4")
    # experiment.generate_model_duo_analysis_df("claude-instant-1.2", "claude-2.1")
    # experiment.generate_model_duo_analysis_df("mistral-tiny", "mistral-medium")
    # check_anthropic_tokens()
    # experiment.clean_and_rerun_analysis()

    # test_prompt_helpers()
    # test_anthropic()
    # test_mistral()
    # prompter()
    # clean_dfs()
    # print("Running Pew Experiment.")

    # experiment = ExperimentPew()
    # experiment.run()
    # experiment.clean()

    experiment = ExperimentOpenEnded()
    experiment.run()

    # test_wasserstein_distance()


def check_anthropic_tokens():
    models = ["claude-instant-1.2", "claude-2.1"]
    for model in models:
        with open(f"./claude/{model}_tokens_in.txt", "r") as file:
            # sum up tokens on each line and
            # print the total tokens used
            tokens = [int(line) for line in file.readlines()]
            print(f"{model} used {sum(tokens)} tokens in.")

        with open(f"./claude/{model}_tokens_out.txt", "r") as file:
            # sum up tokens on each line and
            # print the total tokens used
            tokens = [int(line) for line in file.readlines()]
            print(f"{model} used {sum(tokens)} tokens out.")


def test_wasserstein_distance():
    from scipy.stats import wasserstein_distance

    # Test wasserstein distance with a simple distribution
    u = [0.0, 1.0]
    response1 = [1.0, 0.0]
    response2 = [0.0, 1.0]
    distance1 = wasserstein_distance(u, u, response1, response2)

    print(
        "Distance for a Yes/No question where one model says Yes and the other says No:",
        distance1,
    )

    # Test Yes/Maybe/No
    u = [0.0, 1.0, 2.0]
    response1 = [1.0, 0.0, 0.0]
    response2 = [0.0, 0.0, 1.0]
    distance2 = wasserstein_distance(u, u, response1, response2)

    print(
        "Distance for a Yes/Maybe/No question where one model says Yes and the other says No:",
        distance2,
    )


if __name__ == "__main__":
    main()
