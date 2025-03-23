import os
import sys
import openai
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

from typing import Optional, List
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

# syf 2502不能跑后新加的
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"


Model = Literal["openai/gpt-4", "openai/gpt-3.5-turbo", "openai/gpt-3.5-turbo-instruct", 'qwen-turbo', 'qwen-plus', 'qwen-max']

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion_before2502(prompt: str, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None) -> str:
    # os.environ["http_proxy"] = "127.0.0.1:7890"
    # os.environ["https_proxy"] = "127.0.0.1:7890"

    # gets API Key from environment variable OPENAI_API_KEY
    client = openai.OpenAI(
        # base_url="https://openrouter.ai/api/v1",
        base_url="https://openrouter.ai/",
        api_key=os.getenv('OPENAI_API_KEY')
    )

    response = client.completions.create(
        model='openai/gpt-3.5-turbo-instruct',
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
    )
    # print ("response: ", response)
    return response.choices[0].text


# syf修改新写法： （0203后原来方法好像改了）
# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt: str, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None) -> str:
    # os.environ["http_proxy"] = "127.0.0.1:7890"
    # os.environ["https_proxy"] = "127.0.0.1:7890"

    # syf 2502不能跑后新加的
    os.environ["http_proxy"] = "http://localhost:7890"
    os.environ["https_proxy"] = "http://localhost:7890"

    # gets API Key from environment variable OPENAI_API_KEY
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        # base_url="https://openrouter.ai/",
        api_key=os.getenv('OPENAI_API_KEY')
    )

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    # 原调用写法
    # response = client.completions.create(
    #     model='openai/gpt-3.5-turbo-instruct',
    #     prompt=prompt,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    #     top_p=1,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0,
    #     stop=stop_strs,
    # )
    # # print ("response: ", response)
    # return response.choices[0].text

    # syf 0203后修改
    response = client.chat.completions.create(
        model = 'openai/gpt-3.5-turbo-instruct',  # 原本写法
        # model = 'openai/gpt-4o-mini',
        messages = messages,
        max_tokens=max_tokens,
        stop=stop_strs,
        temperature=temperature,
    )

    return response.choices[0].message.content

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat(prompt: str, model: Model, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None, is_batched: bool = False) -> str:
    assert model != "text-davinci-003"
    # os.environ["http_proxy"] = "127.0.0.1:7890"
    # os.environ["https_proxy"] = "127.0.0.1:7890"

    # syf 2502不能跑后新加的
    os.environ["http_proxy"] = "http://localhost:7890"
    os.environ["https_proxy"] = "http://localhost:7890"

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    # gets API Key from environment variable OPENAI_API_KEY
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv('OPENAI_API_KEY')
        # api_key="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"  # syf跑utils.py测试用
    )

    response = client.chat.completions.create(
        model = model,  # 原本写法
        # model='openai/gpt-4o-mini',  # 0205 syf测试用，因为gpt-3.5-turbo总断 gpt-3.5-turbo-instruct貌似好些
        messages = messages,
        max_tokens=max_tokens,
        stop=stop_strs,
        temperature=temperature,
    )

    return response.choices[0].message.content

if __name__ == '__main__':
    # r = get_chat("say hello world", 'openai/gpt-3.5-turbo')
    r = get_chat("say hello world", 'openai/gpt-4')
    print(r)