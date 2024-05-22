from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch

# 基于本地InternLM自定义LLM类, 将ChatGLM接入到Langchain框架中. 需要从LangChain.llms.base.LLM类继承一个子类, 并重写构造函数与_call函数即可
class ChatGLM_LLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModel = None

    def __init__(self, model_path: str):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True).to(torch.bfloat16).cuda()

        self.model = self.model.eval()
        print("本地模型加载完毕...")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        # 重写调用函数
        response, history = self.model.chat(self.tokenizer, prompt, history=[])

        return response

    @property
    def _llm_type(self) -> str:
        return "ChatGLM-6B"

"""
在上述类定义中，我们分别重写了构造函数和 _call 函数：对于构造函数，我们在对象实例化的一开始加载本地部署的 ChatGLM3-6B 模型，
从而避免每一次调用都需要重新加载模型带来的时间过长；
_call 函数是 LLM 类的核心函数，
LangChain 会调用该函数来调用 LLM,
在该函数中，我们调用已实例化模型的 chat 方法，从而实现对模型的调用并返回调用结果。
"""
