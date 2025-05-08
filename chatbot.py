from camel.models import ModelFactory
from camel.types import ModelType, ModelPlatformType
from camel.configs import ChatGPTConfig, MistralConfig, OllamaConfig

import time

class Chatbot:
    def __init__(self, model_name: str = "gpt"):
        """
        Initialize the chatbot with the specified model.

        Parameters:
            model_name (str): The model to use ("openai" or "mistral").
        """
        self.model_name = model_name.lower()
        self.model = self._get_model()

    def _get_model(self):
        """
        Retrieve the appropriate model based on the user's choice.

        Returns:
            Model: The initialized model.
        """
        if "gpt" in self.model_name:
            return ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
                model_config_dict=ChatGPTConfig().as_dict(),
            )
        elif "mistral" in self.model_name:
            return ModelFactory.create(
                model_platform=ModelPlatformType.MISTRAL,
                model_type=ModelType.MISTRAL_LARGE,
                model_config_dict=MistralConfig(temperature=0.2).as_dict(),
            )
        else:
            return ModelFactory.create(
                model_platform=ModelPlatformType.OLLAMA,
                model_type=self.model_name,
                model_config_dict=OllamaConfig(temperature=0.2, max_tokens=2048).as_dict(),
            )

    def chat(self, user_input: str) -> str:
        """
        Generate a response to the user's input.

        Parameters:
            user_input (str): The user's input message.

        Returns:
            str: The chatbot's response.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ]
        while True:  # Retry loop
            try:
                response = self.model.run(messages)
                return response.choices[0].message.content
            except Exception:
                time.sleep(2)


# Example usage
if __name__ == "__main__":
    # Initialize chatbot with OpenAI model
    chatbot = Chatbot(model_name="openai")
    user_input = "How parasites affect the heart, especially heart failure? Which parasites cause these problems? And how can they do?"
    response = chatbot.chat(user_input)
    print("Chatbot response:", response)