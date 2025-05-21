#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/8/16 08:01
# @Author  : gsunwu@163.com
# @File    : util_baidu.py
# @Description:
# coding=utf-8
import hashlib
import json
import os.path
import pathlib
import traceback
from dataclasses import dataclass
from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.parse import quote_plus


class DemoError(Exception):
    pass


def get_str_md5(input_str: str):
    """
    生成给定字符串的md5

    Parameters
    ----------
    input_str : str

    Returns
    -------
    str
        字符串的md5值
    """
    md5 = hashlib.md5(input_str.encode("utf8"))
    return md5.hexdigest()


@dataclass
class BaiduTTS:
    API_KEY: str = 'peusEXQNel0rPsNRhXpd0aoP'
    SECRET_KEY: str = 'bCIMfqJfEJheSLLqwswzrG7UyqtC53sq'

    TOKEN_URL: str = 'http://aip.baidubce.com/oauth/2.0/token'
    SCOPE: str = 'audio_tts_post'  # 有此scope表示有tts能力，没有请在网页里勾选
    CUID: str = "123456PYTHON"
    TTS_URL: str = 'http://tsn.baidu.com/text2audio'

    def fetch_token(self):
        print("fetch token begin")
        params = {'grant_type': 'client_credentials',
                  'client_id': self.API_KEY,
                  'client_secret': self.SECRET_KEY}
        post_data = urlencode(params)
        post_data = post_data.encode('utf-8')
        req = Request(self.TOKEN_URL, post_data)
        try:
            f = urlopen(req, timeout=5)
            result_str = f.read()
            result = json.loads(result_str)
            print(result)
            if ('access_token' in result.keys() and 'scope' in result.keys()):
                if not self.SCOPE in result['scope'].split(' '):
                    raise DemoError('scope is not correct')
                print(
                    'SUCCESS WITH TOKEN: %s ; EXPIRES IN SECONDS: %s' % (result['access_token'], result['expires_in']))
                return result['access_token']
            else:
                raise DemoError(
                    'MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not found in token response')
        except URLError as err:
            print('token http response http code : ' + str(err))
            traceback.print_exc()

    def speak(self, word):
        """
        输入文字，返回语音
        Parameters
        ----------
        word :

        Returns
        -------

        """

        # 发音人选择, 基础音库：0为度小美，1为度小宇，3为度逍遥，4为度丫丫，
        # 精品音库：5为度小娇，103为度米朵，106为度博文，110为度小童，111为度小萌，默认为度小美
        PER = 0
        # 语速，取值0-15，默认为5中语速
        SPD = 5
        # 音调，取值0-15，默认为5中语调
        PIT = 5
        # 音量，取值0-9，默认为5中音量
        VOL = 5
        # 下载的文件格式, 3：mp3(default) 4： pcm-16k 5： pcm-8k 6. wav
        AUE = 3

        FORMATS = {3: "mp3", 4: "pcm", 5: "pcm", 6: "wav"}
        FORMAT = FORMATS[AUE]

        save_file = pathlib.Path("text_to_audio_cache", f"{get_str_md5(word)}.{FORMAT}")
        # 作缓存
        if os.path.exists(save_file):
            return save_file

        token = self.fetch_token()
        tex = quote_plus(word)  # 此处TEXT需要两次urlencode
        print(tex)
        params = {'tok': token, 'tex': tex, 'per': PER, 'spd': SPD, 'pit': PIT, 'vol': VOL, 'aue': AUE,
                  'cuid': self.CUID,
                  'lan': 'zh', 'ctp': 1}  # lan ctp 固定参数

        data = urlencode(params)
        print('test on Web Browser' + self.TTS_URL + '?' + data)

        req = Request(self.TTS_URL, data.encode('utf-8'))
        try:
            f = urlopen(req)
            result_str = f.read()

            headers = dict((name.lower(), value) for name, value in f.headers.items())
            has_error = ('content-type' not in headers.keys() or headers['content-type'].find('audio/') < 0)
            if has_error:
                raise RuntimeError("error occurred while fetching words")

            if not save_file.parent.exists():
                save_file.parent.mkdir(parents=True)

            with open(save_file, 'wb') as of:
                of.write(result_str)
            if has_error:
                result_str = str(result_str, 'utf-8')
                print("tts api  error:" + result_str)

            return save_file
        except  URLError as err:
            print('asr http response http code : ' + str(err))
