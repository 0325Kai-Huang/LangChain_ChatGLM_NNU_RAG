<h2><center>南师大相关知识库问答助手</center></h2>

**本项目为自己实践动手的RAG小项目，是基于阿里云服务器构建的。相对来说比较简单，很适合新手入门即时获得成就感。**

[阿里云人工智能平台PAI](https://www.aliyun.com/product/bigdata/learn?spm=5176.28536895.J_kUfM_yzYYqU72woCZLHoY.2.4af7586cCxYRdk) 免费试用官方镜像为：py310-cu121-ubuntu22.04, 选择显卡驱动时尽量选较新的支持torch>2.0的

### 环境配置

```python
pip install -r requirements.txt
```

同时，我们需要使用到开源词向量模型[bge-large-zh-v1.5](https://www.modelscope.cn/models/AI-ModelScope/bge-large-zh-v1.5/files) 以及LLM基座模型[ChatGLM3-6B](https://www.modelscope.cn/models/ZhipuAI/ChatGLM-6B/summary)。记住两个模型下载的位置。

### 知识库搭建

可以将知识放在nnu文件夹下面，可以在该文件夹下自定义创建不同大类的关于学校方面的知识，只需要在save_db_data.py文件的下面添加文件夹目录即可

```python
# 目标文件夹  第51行
tar_dir = [
    "/mnt/workspace/nnu/",
]
```

得到所有目标文件路径之后，我们可以使用 LangChain 提供的 FileLoader 对象来加载目标文件，得到由目标文件解析出的纯文本内容。**由于不同类型的文件需要对应不同的 FileLoader，我们判断目标文件类型，并针对性调用对应类型的 FileLoader**.

```python
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import PyPDFLoader
from tqdm import tqdm

def get_text(dir_path):
    # args：dir_path，目标文件夹路径
    # 首先调用上文定义的函数得到目标文件路径列表
    file_lst = get_files(dir_path)
    # docs 存放加载之后的纯文本对象
    docs = []
    # 遍历所有目标文件
    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        print(file_type)
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == 'txt':
            loader = UnstructuredFileLoader(one_file)
        elif file_type == 'pdf':
            loader = PyPDFLoader(one_file)
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
        docs.extend(loader.load())
    return docs
```

LangChain 提供了多种文本分块工具，此处我们使用字符串递归分割器，并选择分块大小为 500，块重叠长度为 20, 接着使用bge开源词向量模型进行本文向量化，同时，我们选择Chroma作为向量数据库，基于上下文分块后的文档以及加载的开源向量化模型，将语料加载到指定路径下的向量数据库：

```python
# 文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=20)
split_docs = text_splitter.split_documents(docs)

# 加载词向量模型
embeddings = HuggingFaceEmbeddings(model_name="你的bge模型地址")

# 语料加载到指定路径下的向量数据库
# 构建向量数据库
# 定义持久化路径
persist_directory = '你想保存的位置'
# 加载数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
# 将加载的向量数据库持久化到磁盘上
vectordb.persist()
```

运行save_db_data.py文件即可完成本地向量数据库构建，后续直接导入该数据库即可，无需重复构建。

### ChatGLM-6B接入Langchain

我们需要基于本地部署的 ChatGLM3-6B，自定义一个 LLM 类，将 ChatGLM 接入到 LangChain 框架中。完成自定义 LLM 类之后，可以以完全一致的方式调用 LangChain 的接口，而无需考虑底层模型调用的不一致。

基于本地部署的 ChatGLM3-6B 自定义 LLM 类并不复杂，我们只需从 LangChain.llms.base.LLM 类继承一个子类，并重写构造函数与 `_call` 函数即可：

保存到LLM.py文件中，注意的是不要使用transformers库中的AutoModelForCausalLM来构建。

```python
self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
```

我在使用AutoModelForCausalLM构建时报错：

```
ValueError: Unrecognized configuration class <class 'transformers_modules.chatglm6b.configuration_chatglm.ChatGLMConfig'> for this kind of AutoModel: AutoModelForCausalLM.
```

查看下载的ChatGLM3-6B模型目录中的config.json文件里auto_map字段下没有AutoModelForCausalLM键值对，因此我改成了AutoModel来构建就没有上述报错。

```json
{
  "_name_or_path": "./chatglm-6b",
  "architectures": [
    "ChatGLMModel"
  ],
  "auto_map": {
    "AutoConfig": "configuration_chatglm.ChatGLMConfig",
    "AutoModel": "modeling_chatglm.ChatGLMForConditionalGeneration",
    "AutoModelForSeq2SeqLM": "modeling_chatglm.ChatGLMForConditionalGeneration"
  },
  "bos_token_id": 130004,
  "eos_token_id": 130005,
  "mask_token_id": 130000,
  "gmask_token_id": 130001,
  "pad_token_id": 3,
  "hidden_size": 4096,
  "inner_hidden_size": 16384,
  "layernorm_epsilon": 1e-05,
  "max_sequence_length": 2048,
  "model_type": "chatglm",
  "num_attention_heads": 32,
  "num_layers": 28,
  "position_encoding_2d": true,
  "torch_dtype": "float16",
  "transformers_version": "4.23.1",
  "use_cache": true,
  "vocab_size": 130528
}
```

### 部署Web Demo

基于 Gradio 框架将其部署到 Web 网页，从而搭建一个小型 Demo，便于测试与使用。可以通过执行nnu_chat_rag_web_demo.py即可在本地启动知识库助手。

#### 阿里云服务器构建踩坑

如果出现进行gradio页面后输入问题，点击chat按钮一直计时，如下图：

![requests_time_out](images\requests_time_out.png)

可以重新刷新页面，按F12查看浏览器过程，重新输入问题，然后点击chat按钮，查看请求中是否有403forbidden。

![403_forbidden](images\403_forbidden.png)

如果有上述问题，可以通过构建公共gradio的url来访问，操作如下:

```python
#1. 下载文件, 如果被安全中心拦截，先暂时关闭安全中心
https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64

#2. 把文件重命名为 frpc_linux_amd64_v0.2

#3. 把该文件移动到/opt/conda/lib/python3.10/site-packages/gradio
# 上面是服务器的当前python环境，如果你是创建的虚拟python环境，则移动到
# /opt/conda/envs/your_env_name/lib/python3.10/site-packages/gradio
# 并添加可执行权力
chmod +x frpc_linux_amd64_v0.2

# 并且将nnu_chat_rag_web_demo.py中的launch()方法更改为
demo.launch(server_name="127.0.0.1", server_port=8501, inbrowser=True, share=True)
```

这个时候命令行中会给你创建公共链接

![public_link](images\public_link.png)

点击公共链接，输入问题，即可成功！！！

![success](images\success.png)