#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/9/17 12:14
# @Author  : gsunwu@163.com
# @File    : util_chattts.py
# @Description:
import hashlib
import os
import re
import time
import wave
from enum import Enum
import numpy as np
import warnings
import pyttsx3

warnings.filterwarnings("ignore")
import abc


def has_chinese(text):
    # 使用正则表达式匹配中文字符
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(text))


# class TTS(metaclass=abc.ABCMeta):
#
#     @abc.abstractmethod
#     def say_chines(self):
#         """
#         Get the training data without labels
#
#         Returns
#         -------
#
#         """
#         pass

class ChatTTS:
    chat = None

    @staticmethod
    def say_texts(texts: str = None, file_name=None, seed=3):

        if texts is None:
            texts = [
                "我爱中国，I love china",
            ]

        if file_name is None:
            file_name = "test.aiff"

        if not file_name.endswith(".aiff"):
            file_name += ".aiff"

        if os.path.exists(file_name):
            print(f"文件已存在：{os.path.abspath(file_name)}")
            return file_name

        chat = ChatTTS.InitChat()
        r = chat.sample_random_speaker(seed=seed)
        params_infer_code = {
            "spk_emb": r,  # add sampled speaker
            "temperature": 0.3,  # using custom temperature
            "top_P": 0.7,  # top P decode
            "top_K": 20,  # top K decode
        }
        wavs = chat.infer(texts, use_decoder=True, params_infer_code=params_infer_code)

        audio_data = np.array(wavs[0], dtype=np.float32)
        sample_rate = 24000
        audio_data = (audio_data * 32767).astype(np.int16)

        with wave.open(file_name, "w") as wf:
            wf.setnchannels(1)  # Mono channel
            wf.setsampwidth(2)  # 2 bytes per sample
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

        print("音频保存为：", os.path.abspath(file_name))
        return file_name

    @classmethod
    def InitChat(cls):
        from ChatTTS import Chat
        if ChatTTS.chat is None:
            print("加载模型中...")
            # chattts==0.1.1
            chat = Chat()
            # download from: https://huggingface.co/2Noise/ChatTTS/tree/main
            # (base) ➜  ~ git:(master) ✗ ls /Users/sunwu/Documents/chat_tts_models
            # asset  config
            chat.load_models(source="local", local_path="/Users/sunwu/Documents/chat_tts_models")
            ChatTTS.chat = chat

        return ChatTTS.chat


class TypeVoice(Enum):
    # Chinese = "com.apple.voice.premium.zh-CN.Lilian"
    # English = "com.apple.voice.compact.en-US.Samantha"
    # English2 = "com.apple.voice.enhanced.en-US.Noelle"
    Chinese = "Lilian (Premium)"
    English = "Siri Voice 4 (Enhanced)"
    English2 = "Siri Voice 5 (Enhanced)"


import subprocess
import sys


def execute_command(command):
    if command.find("nohup") > -1:
        if command.find("2>&1") == -1:
            raise ValueError("When nohup is existed, 2>&1 must exist. For example >/dev/null 2>&1")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    while True:

        try:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                msg = output.strip()
                print("---- " + msg.decode("utf-8") + "\n")
        except:
            traceback.print_exc()
    rc = process.poll()
    if rc != 0:
        print("---- Exec command error: ", process.stderr.readlines(), file=sys.stderr)
        return None
    # process.stdin.close()
    process.stdout.close()
    process.stderr.close()
    return None


def _py_tts_say(text: str, output_file: str = "output1.aiff", voice_id=TypeVoice.English.value):
    """
    中文：
    Parameters
    ----------
    text :
    output_file :
    voice_id :

    Returns
    -------

    """
    # 示例：使用 say 命令
    command = ['say', '-v', f'"{voice_id}"', f'"{text}"', '-o', output_file]
    _run_cmd = " ".join(command)
    execute_command(_run_cmd)
    return output_file
    # """
    # say -v Alex "I will speak this text. 我将说这段话" -o output.aiff
    # say -v "?"
    # """
    # # 防止一个字都没有时卡住，所以加一个逗号
    # if text == "" or text is None:
    #     raise RuntimeError("Say text cant be empty")
    #
    # # 初始化语音引擎
    # engine = pyttsx3.init()
    #
    # # 获取可用的语音列表
    # voices = engine.getProperty('voices')
    #
    # # 是否发现语音
    # voice_found = False
    #
    # for voice in voices:
    #     if voice.id == voice_id:
    #         engine.setProperty('voice', voice.id)
    #         voice_found = True
    #         break
    # if not voice_found:
    #     print(f"未找到语音: {voice_id}")
    #     raise RuntimeWarning(f"未找到语音: {voice_id},请在VoiceOver Utility 中下载")
    # else:
    #     print(f"正在使用语音: {voice_id}")
    #
    # # 设置要说的文本
    # # text = "I will speak this text。我将说这段话"
    #
    # # 保存为 .aiff 文件
    # # 使用 save_to_file 方法保存音频
    # engine.save_to_file(text, output_file)
    # # 等待语音播放完成
    # engine.runAndWait()
    # print(f"语音已保存为 {output_file}")
    # return output_file


def get_str_hash(string):
    """
    Get the hash code of string

    Parameters
    ----------
    string : str
        The input str

    Returns
    -------
    str
        The hash code of the string

    """
    if not isinstance(string, str):
        raise ValueError(f"get_str_hash excepts str, but recept {type(string)}")
    return hashlib.sha1(string.encode("utf8")).hexdigest()


class PyTTS:
    chat = None

    @staticmethod
    def say_english(text: str, file_name=None, voice_id: str = "Tom", cache=True,
                    home="runtime/"):
        file_name = PyTTS.verify_file_name(file_name)
        if os.path.exists(file_name) and cache is True:
            print(f"文件已存在：{os.path.abspath(file_name)}")
            return file_name

        _py_tts_say(text, output_file=file_name, voice_id=voice_id)
        print("音频保存为：", os.path.abspath(file_name))
        return file_name

    @staticmethod
    def say_english2(text: str, cache=True):
        """
        使用语调2发音
        Parameters
        ----------
        text :
        file_name :
        cache :

        Returns
        -------

        """
        file_name = PyTTS.get_text_file_name(text, label="_en2")
        return PyTTS.say_english(text, file_name, voice_id=TypeVoice.English2.value, cache=cache)

    @staticmethod
    def say_english1(text: str, cache=True, file_name=None):
        """
        使用语调1发音
        Parameters
        ----------
        text :
        file_name :
        cache :

        Returns
        -------

        """
        assert len(text) >=1
        if file_name is None:
            file_name = PyTTS.get_text_file_name(text, label="_en1")
        return PyTTS.say_english(text, file_name, voice_id=TypeVoice.English.value, cache=cache)

    @staticmethod
    def verify_file_name(file_name, ext=".aiff"):
        """
        预处理输入文件，并输出可用的文件
        Parameters 
        ----------
        file_name :

        Returns
        -------

        """
        if file_name is None:
            file_name = f"test{ext}"
        if not file_name.endswith(ext):
            file_name += ext
        return file_name

    @staticmethod
    def get_text_file_name(text, label="", ext=".aiff", home="./runtime"):
        """
        预处理输入文件，并输出可用的文件
        Parameters
        ----------

        Returns
        -------

        """
        file_name = f"{home}/{get_str_hash(text)}{label}{ext}"
        if not file_name.endswith(ext):
            file_name += ext
        return os.path.abspath(file_name)

    @staticmethod
    def say_texts(text: str, file_name=None):
        # file_name = PyTTS.verify_file_name(file_name)
        #
        # if os.path.exists(file_name):
        #     print(f"文件已存在：{os.path.abspath(file_name)}")
        #     return file_name
        #
        # _py_tts_say(text, output_file=file_name)
        # print("音频保存为：", os.path.abspath(file_name))
        # return file_name
        if file_name is None:
            file_name = os.path.join("runtime", get_str_hash(text))
        if has_chinese(text):
            return PyTTS.say_chinese(text, file_name)
        else:
            return PyTTS.say_english(text, file_name)

    @classmethod
    def InitChat(cls):
        pass

    @staticmethod
    def list_all_voices():
        ids = []
        import pyttsx3
        # 初始化语音引擎
        engine = pyttsx3.init()

        # 获取可用的语音列表
        voices = engine.getProperty('voices')

        # 列出所有支持的语音
        for voice in voices:
            print(f"ID: {voice.id}")
            print(f"Name: {voice.name}")
            print(f"Languages: {voice.languages}")
            print(f"Gender: {voice.gender}")
            print(f"Age: {voice.age}")
            print("-" * 40)
            ids.append(voice.id)
        return ids

    @staticmethod
    def say_chinese(text: str, file_name=None, voice_id: str = "com.apple.voice.premium.zh-CN.Lili"):
        file_name = PyTTS.verify_file_name(file_name)
        if os.path.exists(file_name):
            print(f"文件已存在：{os.path.abspath(file_name)}")
            return file_name

        _py_tts_say(text, output_file=file_name, voice_id=voice_id)
        print("音频保存为：", os.path.abspath(file_name))
        return file_name


def test_all_voice():
    for id in PyTTS.list_all_voices():
        # _py_tts_say("Hello, i am siri. ", output_file=id + ".aiff", voice_id=id)
        _py_tts_say("你好，我是siri", output_file=id + ".aiff", voice_id=id)


if __name__ == '__main__':
    PyTTS.list_all_voices()
    # PyTTS.say_texts("i am siri.")

    PyTTS.say_english("i am siri", "voice_en", TypeVoice.Chinese, cache=False)
    # PyTTS.say_chinese("你好，我是siri", "voice_zh")
    # test_all_voice()
