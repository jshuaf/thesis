import concurrent.futures
import math
from helpers import *

from rich.console import Console
from rich.prompt import Prompt
from vertexai.language_models import TextGenerationModel
import os
from dotenv import load_dotenv
from tenacity import retry, wait_fixed, stop_after_attempt
import json
import time

load_dotenv(override=True)  # take environment variables from .env.

class ModelClient:

    def __init__(
        self,
        vertex=False,
        openAI=False,
        anthropic=False,
        mistral=False,
        azure=False,
        modifier=None,
        logprobs=None,
        temperature=1,
        top_p=1,
        top_k=40,
        max_tokens=100,
        vertex_model="text-bison@002",
    ):
        if openAI:
            from openai import OpenAI

            self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        if vertex:
            import google.generativeai as gemini

            self.gemini_client = gemini.GenerativeModel("gemini-pro")
            gemini.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            # vertexai.init(project="model-constitutions2", location="us-central1")
        if anthropic:
            from anthropic import Anthropic

            self.anthropic_client = Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        if mistral:
            from mistralai.client import MistralClient

            self.mistral_client = MistralClient(
                api_key=os.environ.get("MISTRAL_API_KEY")
            )

        self.console = Console()
        self.modifier = modifier
        self.logprobs = logprobs
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.vertex_model_predictor = TextGenerationModel.from_pretrained(vertex_model)

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(3), reraise=True)
    def call_model(self, prompt: str, model: str) -> str:
        api = self._get_api(model)
        if self.modifier:
            prompt = self.modifier.format(prompt)

        if api == ModelAPI.OPENAI:
            return self._call_openai_model(prompt, model)
        elif api == ModelAPI.GEMINI:
            return self._call_gemini_model(prompt)
        elif api == ModelAPI.VERTEX:
            return self._call_vertex_model(prompt, model)
        elif api == ModelAPI.ANTHROPIC:
            return self._call_anthropic_model(prompt, model)
        elif api == ModelAPI.MISTRAL:
            return self._call_mistral_model(prompt, model)
        elif api == ModelAPI.AZURE:
            return self._call_azure_model(prompt, model)

    def _get_api(self, model: str) -> ModelAPI:
        return MODELS_TO_APIS[model]

    def _call_openai_model(self, prompt: str, model: str) -> str:
        chat_completion = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            logprobs=(not not self.logprobs),
            top_logprobs=self.logprobs,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=10,
        )
        if self.logprobs:
            logprobs = chat_completion.choices[0].logprobs.content[0].top_logprobs
            # exponentiate and normalize the logprobs
            exp_logprobs = {x.token: math.exp(x.logprob) for x in logprobs}
            normalized_probs = {
                k: v / sum(exp_logprobs.values()) for k, v in exp_logprobs.items()
            }
            return normalized_probs
        return chat_completion.choices[0].message.content

    def _call_gemini_model(self, prompt: str) -> str:
        response = self.gemini_client.generate_content(prompt)
        return response.text

    def _call_vertex_model(self, prompt: str, model: str) -> str:
        parameters = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        if self.logprobs:
            parameters["logprobs"] = self.logprobs
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.vertex_model_predictor.predict, prompt, **parameters
                )
                response = future.result(timeout=10)  # Enforces a 10-second timeout
                if self.logprobs:
                    raw_logprobs = response._prediction_response.predictions[0][
                        "logprobs"
                    ]["topLogProbs"][0]
                    exp_logprobs = {k: math.exp(v) for k, v in raw_logprobs.items()}
                    normalized_probs = {
                        k: v / sum(exp_logprobs.values())
                        for k, v in exp_logprobs.items()
                    }

                    return normalized_probs
                return response.text
        except concurrent.futures.TimeoutError as e:
            raise TimeoutError("Model prediction timeout") from e

    def _call_replicate_model(self, prompt: str, model: str) -> str:
        import replicate

        modelFullNames = {
            "llama2-70b": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            "llama2-13b": "meta/llama-2-13b-chat:56acad22679f6b95d6e45c78309a2b50a670d5ed29a37dd73d182e89772c02f1",
            "mistral": "mistralai/mistral-7b-instruct-v0.2:f5701ad84de5715051cb99d550539719f8a7fbcf65e0e62a3d1eb3f94720764e",
        }
        modelName = modelFullNames[model]
        output = replicate.run(
            modelName, input={"prompt": prompt, "max_new_tokens": 256}
        )
        output = replicate.run(model, input={"prompt": prompt})
        return "".join(output)

    def _call_anthropic_model(self, prompt: str, model: str) -> str:
        # Add prompt to ./claude/{model}.txt with a new line
        # And create file if it doesn't exist
        from anthropic import HUMAN_PROMPT, AI_PROMPT

        with open(f"./claude/{model}_in.txt", "a") as file:
            file.write(f"{prompt}\n")

        # Track total tokens in separate file - add to token tracker
        input_tokens = self.anthropic_client.count_tokens(prompt)
        with open(f"./claude/{model}_tokens_in.txt", "a") as file:
            file.write(f"{input_tokens}\n")

        completion = self.anthropic_client.completions.create(
            model=model,
            max_tokens_to_sample=self.max_tokens,
            prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        output = completion.completion
        # Track total tokens in separate file - add to token tracker
        output_tokens = self.anthropic_client.count_tokens(output)
        with open(f"./claude/{model}_tokens_out.txt", "a") as file:
            file.write(f"{output_tokens}\n")
        with open(f"./claude/{model}_out.txt", "a") as file:
            file.write(f"{output}\n")

        return output

    def _call_mistral_model(self, prompt: str, model: str) -> str:
        from mistralai.models.chat_completion import ChatMessage

        messages = [ChatMessage(role="user", content=prompt)]
        # No streaming
        chat_response = self.mistral_client.chat(
            model=model,
            messages=messages,
            top_p=self.top_p,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return chat_response.choices[0].message.content

    def _call_azure_model(self, prompt: str, model: str) -> str:
        import urllib.request
        import urllib.error

        data = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        body = str.encode(json.dumps(data))

        models_to_urls = {
            "llama-2-70b-chat": "https://Llama-2-70b-chat-thesis-serverless.eastus2.inference.ai.azure.com/v1/chat/completions",
            "llama-2-7b-chat": "https://Llama-2-7b-chat-thesis-1-serverless.eastus2.inference.ai.azure.com/v1/chat/completions",
        }
        models_to_api_keys = {
            "llama-2-70b-chat": os.environ.get("LLAMA_70B_API_KEY"),
            "llama-2-7b-chat": os.environ.get("LLAMA_7B_API_KEY"),
        }
        url = models_to_urls.get(model)
        if not url:
            return
        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + models_to_api_keys.get(model)),
        }
        req = urllib.request.Request(url, body, headers)

        try:
            response = urllib.request.urlopen(req)
            result = response.read()
            return json.loads(result.decode("utf-8"))["choices"][0]["message"][
                "content"
            ]
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))
            print(error.info())
            print(error.read().decode("utf8", "ignore"))


def main():
    clientYN = ModelClient(
        openAI=True,
        vertex=True,
        mistral=True,
        azure=True,
        max_tokens=30,
        anthropic=True,
        modifier=MODIFIERS["yesno"],
    )
    client = ModelClient(
        openAI=True,
        vertex=True,
        mistral=True,
        azure=True,
        max_tokens=30,
        anthropic=True,
    )
    console = Console()
    while True:
        console.print(
            "[bold green]Enter your prompt (or type 'exit' to quit):[/bold green]"
        )
        prompt = Prompt.ask("Prompt")

        if prompt.lower() == "exit":
            break

        models = [
            "gpt-3.5-turbo-0125",
            "gpt-4",
            "mistral-tiny",
            "llama-2-70b-chat",
            "claude-2",
        ]
        if "[yn]" in prompt:
            currentClient = clientYN
        else:
            currentClient = client
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_metadata = {
                executor.submit(currentClient.call_model, prompt, model): model
                for model in models
            }
            for future in concurrent.futures.as_completed(future_to_metadata):
                response = future.result()
                model = future_to_metadata[future]
                color = MODELS_TO_COLORS.get(model, "white")
                console.print(
                    f"\n[bold {color}] {model} Response:[/bold {color}]",
                    response,
                )

        console.print("\n---------------------------------------\n")


if __name__ == "__main__":
    main()
