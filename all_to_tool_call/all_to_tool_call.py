from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests
import argparse
from typing import Optional, List
from openai import OpenAI
# 创建FastAPI应用实例
app = FastAPI()

class ChatCompletionRequest(BaseModel):
    
    model: str
    messages: List
    max_tokens: int = 4096
    temperature: float = 0.7
    tools: Optional[List] = None

parser = argparse.ArgumentParser(description="启动模型服务代理")
parser.add_argument('--no_tool_call_base_url', type=str, default="https://api.siliconflow.cn/v1")
parser.add_argument('--no_tool_call_model_name', type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct")
parser.add_argument('--no_tool_call_api_key', type=str, default="sk-")
parser.add_argument('--tool_call_base_url', type=str, default="https://api.siliconflow.cn/v1")
parser.add_argument('--tool_call_model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument('--tool_call_api_key', type=str, default="sk-")
parser.add_argument('--host', type=str, default="0.0.0.0")
parser.add_argument('--port', type=int, default=8888)
args = parser.parse_args()

def generate_text(base_url: str, model: str, messages: List, max_tokens: int, temperature: float, api_key: str, tools=None):

    client = OpenAI(base_url=base_url, api_key=api_key)



    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
        tools=tools,
    )
    return completion


# 定义路由和处理函数，与OpenAI API兼容
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    
    if request.tools:
        TOOL_EAXMPLE = "You will receive a JSON string containing a list of callable tools. Please parse this JSON string and return a JSON object containing the tool name and tool parameters."
 
        REUTRN_FORMAT="{\"tool\": \"tool name\", \"parameters\": {\"parameter name\": \"parameter value\"}}"
        
        INSTRUCTION = f"""
        {TOOL_EAXMPLE}
        Answer the following questions as best you can. 
                
        Use the following format:
        ```tool_json
        {REUTRN_FORMAT}
        ``` 
        
        Please choose the appropriate tool according to the user's question. If you don't need to call it, please reply directly to the user's question. When the user communicates with you in a language other than English, you need to communicate with the user in the same language.
        
        When you have enough information from the tool results, respond directly to the user with a text message without having to call the tool again.
        
        You can use the following tools:
        {request.tools}
        """
        messages = [{"role": "system", "content": INSTRUCTION}]
        messages +=  request.messages
        response = generate_text(args.no_tool_call_base_url, args.no_tool_call_model_name, messages, request.max_tokens, request.temperature, args.no_tool_call_api_key)
        response = response.choices[0].message.content
        print(response)
        messages = [{"role": "system", "content": "Answer the initial <QUESTION> based on the <INFORMATION> directly."}]
        print(request.messages[-1]['content'])
        messages += [{"role": "user", "content": f"<QUESTION>\n{request.messages[-1]['content']}\n</QUESTION>\n<INFORMATION>\n{response}\n</INFORMATION>"}
        ]
        response = generate_text(args.tool_call_base_url, args.tool_call_model_name, messages, request.max_tokens, request.temperature, args.tool_call_api_key, tools=request.tools)
        
    else:
        response = generate_text(args.no_tool_call_base_url, args.no_tool_call_model_name, request.messages, request.max_tokens, request.temperature, args.no_tool_call_api_key)
    
    return response

# 启动FastAPI应用，使用命令行参数指定的端口
if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)