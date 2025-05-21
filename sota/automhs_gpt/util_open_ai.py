#!/usr/bin/python3
# _*_ coding: utf-8 _*_
import openai
import rich
from openai import OpenAI
from openai.types.chat import ChatCompletion


# gpt-4o-mini-2024-07-18
class UtilOpenai:

    @staticmethod
    def get_client():
        client = openai.AzureOpenAI(
            api_key="562eacb19033471daa05b3594c790492",
            azure_endpoint="https://arktao.openai.azure.com",
            api_version="2023-05-15",
            azure_deployment="arktao"
        )
        return client

    @staticmethod
    def chat_gpt35(message: list, stream=False) -> ChatCompletion:
        """

        Parameters
        ----------
        message : list of objects with role and content keys.
             messages = [
                {"role": "system", "content": "You are a helpful AI assistant that translates English to Chinese."},
                {"role": "user", "content": f"Please translate the following text to Chinese:\n\nI love china."}
            ]
        stream : bool


        Returns
        -------

        """
        client = UtilOpenai.get_client()
        response = client.chat.completions.create(
            model="gpt-35-turbo-16k",  # 使用部署ID作为引擎名称
            messages=message,
            stream=stream
        )
        rich.print("token expense: ", response.usage)
        return response

    @staticmethod
    def chat_gpt4_turbo(message):
        client = OpenAI(
            base_url="https://api.gptsapi.net/v1",
            api_key="sk-qeEfcc169623773ee4d891edd4184615cbf906d70daFLkic"
        )

        completion = client.chat.completions.create(
            extra_headers={
            },
            model="gpt-4o-2024-05-13",
            messages=message,
        )
        return completion

    @staticmethod
    def chat_gpt4o_mini_2024_07_18(message):
        client = OpenAI(
            base_url="https://api.gptsapi.net/v1",
            api_key="sk-qeEfcc169623773ee4d891edd4184615cbf906d70daFLkic"
        )

        completion = client.chat.completions.create(
            extra_headers={
            },
            model="gpt-4o-mini-2024-07-18",
            messages=message,
        )
        return completion

    @staticmethod
    def chat_by_modelname(message, model_name="gpt-4o-mini-2024-07-18"):
        client = OpenAI(
            base_url="https://api.gptsapi.net/v1",
            api_key="sk-qeEfcc169623773ee4d891edd4184615cbf906d70daFLkic"
        )

        completion = client.chat.completions.create(
            extra_headers={
            },
            model=model_name,
            messages=message,
        )
        return completion

    @classmethod
    def parse_message_without_stream(cls, msg: ChatCompletion):
        return msg.choices[0].message.content


if __name__ == '__main__':
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant that translates English to Chinese."},
        {"role": "user", "content": f"Please translate the following text to Chinese:\n\nI love china."}
    ]
    msg = UtilOpenai.chat_gpt4o_mini_2024_07_18(messages)

    print(UtilOpenai.parse_message_without_stream(msg))
