import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Optional
from langchain_community.embeddings import OpenVINOBgeEmbeddings
import click
import uvicorn
import tiktoken
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from langchain_community.embeddings import OpenVINOBgeEmbeddings
from pydantic import BaseModel

encoder = tiktoken.get_encoding("cl100k_base")

class EmbeddingRequest(BaseModel):
    #输入可以是字符串、字符串列表、至于为什么要加上List[List[int]]，因为在集成maxkb的过程中发现，其调用向量模型传过来的参数中文字是经过tiktoken编码的。
    input: str|List[str]|List[List[int]]
    model: str
    
TIMEOUT_KEEP_ALIVE = 5  # seconds.

class OpenaiServer:

    def __init__(self,
                 embedding_model_path):
        
        # 可在此处修改为自己的模型，可以通过任意方式加载（huggingface，langchain，sentence-transformers等）,
        # 此处为了加速使用了openvino模型，可根据需要自行修改为自己的模型
        self.model = OpenVINOBgeEmbeddings(
        model_name_or_path=embedding_model_path,
        model_kwargs={"device": "CPU"},
        encode_kwargs={"normalize_embeddings": True},
)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # terminate rank0 worker
            yield

        self.app = FastAPI(lifespan=lifespan)


        self.register_routes()
    def register_routes(self):
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/v1/embeddings",
                               self.get_embeddings,
                               methods=["POST"])

    async def health(self) -> Response:
        return Response(status_code=200)
    async def get_embeddings(self, request: EmbeddingRequest) -> Response:
        
        data = [] 
        if isinstance(request.input, List):
            if isinstance(request.input[0], str):
                # 修改完模型后，可根据向量模型的具体推理方式修改如下方法
                # 此处为langchain加载的向量模型所使用的推理方法embed_documents和embed_query，分别针对列表和字符串
                # 如使用sentence_transformers,推理方法如下：
                # from sentence_transformers import SentenceTransformer
                # model = SentenceTransformer("shibing624/text2vec-base-chinese")
                # sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
                # sentence_embeddings = model.encode(sentences)
                
                embedding = self.model.embed_documents(request.input)
                for i, item in enumerate(embedding):
                    tmp = {
                        "object": "embedding",
                        "embedding": item,
                        "index": i
                            }
                    data.append(tmp)    
            elif isinstance(request.input[0], List):
                # 将tiktoken编码的文本转会文本
                text_list = [encoder.decode(item) for item in request.input]
                embedding = self.model.embed_documents(text_list)
                for i, item in enumerate(embedding):
                    tmp = {
                        "object": "embedding",
                        "embedding": item,
                        "index": i
                            }
                    data.append(tmp)
        else:
            # 
            embedding = self.model.embed_query(request.input)
            tmp = {
            "object": "embedding",
            "embedding": embedding,
            "index": 0
                }
            data.append(tmp)
        
        
        res = {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0
            }
            }
        return JSONResponse(content=res)

    async def __call__(self, host, port):
        config = uvicorn.Config(self.app,
                                host=host,
                                port=port,
                                log_level="info",
                                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        await uvicorn.Server(config).serve()
@click.command()
@click.argument("model_dir")
@click.option("--host", type=str, default=None)
@click.option("--port", type=int, default=8000)
def entrypoint(model_dir,
               host: Optional[str] = None,
               port: int = 8000):
    host = host or "0.0.0.0"
    port = port or 8000
    logging.info(f"Starting server at {host}:{port}")

    server = OpenaiServer(embedding_model_path=model_dir)

    asyncio.run(server(host, port))

if __name__ == "__main__":
    entrypoint()
    
    
    
    