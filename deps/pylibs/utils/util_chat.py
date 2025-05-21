import dataclasses
import pprint
import re
from enum import Enum
from openai import OpenAI
import os


@dataclasses.dataclass
class ChatStatus:
    model: str
    finish_reason: str
    created: int
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class UtilEnv:
    @staticmethod
    def get_env(key) -> str:
        env_val = os.getenv(key, default=None)
        if env_val is None:
            raise RuntimeError(f"Environment {key} is not set")
        else:
            return env_val


def clean_text(text):
    """
    对输入的文本进行清理和规范化处理
    Parameters
    ----------
    text :

    Returns
    -------

    """
    # Remove control characters
    text = re.sub(r'[\r\n\t]', '', text)

    # Strip leading and trailing whitespace
    text = text.strip()

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    return text


# Example usage

class ChatRole(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


"""
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "I'm not sure, but you can check a weather website for the latest updates."}
]
"""


@dataclasses.dataclass
class ChatBot:
    base_url: str
    api_key: str
    model: str
    dry_run: bool = True
    delimiter: str = "===="
    sys_prompt: str = None

    def __post_init__(self):
        self.conversation_history_ = []
        if self.sys_prompt is None:
            self.sys_prompt = f"""
            Please check the text (maybe in latex) I gave you for grammatical errors or spelling mistakes. If there are, output the specific errors, then tell me the specific reason, finally give me the improvements. If not, output no, do not output any other text. 
            You should think step by step. The text I gave you will be delimited with {self.delimiter} characters
            """
        self.add_system_message(self.sys_prompt)
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def add_user_message(self, content):
        """
         {"role": "system",
         "content": "Please check the text I gave you for grammatical errors or spelling mistakes. If there are, output the specific errors, then tell me why, finally give me the improvements. If not, output no, do not output an explanation. You should think step by step. The text I gave you is wrapped in ===."},
        {"role": "user", "content": f"===\n{content}\n==="}

        """

        self.conversation_history_.append(
            {"role": ChatRole.user.value, "content": f"{self.delimiter}{content}{self.delimiter}"})
        # Keep only the last 3 messages
        if len(self.conversation_history_) > 3:
            self.conversation_history_.pop(1)

    def add_system_message(self, content):
        """
         {"role": "system",
         "content": "Please check the text I gave you for grammatical errors or spelling mistakes. If there are, output the specific errors, then tell me why, finally give me the improvements. If not, output no, do not output an explanation. You should think step by step. The text I gave you is wrapped in ===."},
        {"role": "user", "content": f"===\n{content}\n==="}

        """
        content = clean_text(content)
        self.conversation_history_.append({"role": ChatRole.system.value, "content": f"{content}"})

    def add_assistant_message(self, content):
        """
         {"role": "system",
         "content": "Please check the text I gave you for grammatical errors or spelling mistakes. If there are, output the specific errors, then tell me why, finally give me the improvements. If not, output no, do not output an explanation. You should think step by step. The text I gave you is wrapped in ===."},
        {"role": "user", "content": f"===\n{content}\n==="}

        """

        self.conversation_history_.append(
            {"role": ChatRole.assistant.value, "content": f"{self.delimiter}{content}{self.delimiter}"})
        # Keep only the last 3 messages
        if len(self.conversation_history_) > 3:
            self.conversation_history_.pop(1)

    def add_message(self, role, content):
        """
         {"role": "system",
         "content": "Please check the text I gave you for grammatical errors or spelling mistakes. If there are, output the specific errors, then tell me why, finally give me the improvements. If not, output no, do not output an explanation. You should think step by step. The text I gave you is wrapped in ===."},
        {"role": "user", "content": f"===\n{content}\n==="}
        """
        content = clean_text(content)
        self.conversation_history_.append({"role": role, "content": f"{self.delimiter}{content}{self.delimiter}"})
        # Keep only the last 3 messages
        if len(self.conversation_history_) > 3:
            self.conversation_history_.pop(1)

    def chat(self, user_message):
        # Add user's message to the history
        self.add_user_message(user_message)

        if not self.dry_run:
            # Send the conversation history to the API
            pprint.pprint("conversation_history_: ")
            pprint.pprint(self.conversation_history_)
            completion = self.client.chat.completions.create(
                extra_headers={},
                model=self.model,
                messages=self.conversation_history_
            )

            # Extract the assistant's response
            top_choice = completion.choices[0]
            assistant_message = top_choice.message.content
            chat_statics = ChatStatus(model=completion.model,
                                      created=completion.created,
                                      completion_tokens=completion.usage.completion_tokens,
                                      prompt_tokens=completion.usage.prompt_tokens,
                                      total_tokens=completion.usage.total_tokens,
                                      finish_reason=top_choice.finish_reason,
                                      )
            pprint.pprint(chat_statics)

        else:
            assistant_message = "Dru run message"
        # Add assistant's response to the history
        self.add_assistant_message(assistant_message)

        return assistant_message


# Example usage
# "gpt-4o-mini-2024-07-18"
if __name__ == '__main__':

    base_url = UtilEnv.get_env("OPEN_AI_URL")
    api_key = UtilEnv.get_env("OPEN_AI_KEY")
    model = UtilEnv.get_env("OPEN_AI_MODEL_NAME")

    chat_bot = ChatBot(base_url=base_url, api_key=api_key, model=model, dry_run=False)
    # Example of multiple calls
    user_messages = [
        "I am sunwu",
        "Who i am?",
    ]

    for message in user_messages:
        response = chat_bot.chat(message)
        print(f"User: {message}")
        print(f"Assistant: {response}")
