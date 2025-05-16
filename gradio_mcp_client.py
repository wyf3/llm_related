import gradio as gr
from mcp.client.sse import sse_client
from mcp import ClientSession
from openai import AsyncOpenAI
import json


SYSTEM_PROMPT = """你是一个AI助手。
你可以使用 MCP 服务器提供的工具来完成任务。
MCP 服务器会动态提供工具，你需要先检查当前可用的工具。

在使用 MCP 工具时，请遵循以下步骤：
1、根据任务需求选择合适的工具
2、按照工具的参数要求提供正确的参数
3、观察工具的返回结果，并根据结果决定下一步操作
4、工具可能会发生变化，比如新增工具或现有工具消失

请遵循以下指南：
- 使用工具时，确保参数符合工具的文档要求
- 如果出现错误，请理解错误原因并尝试用修正后的参数重新调用
- 按照任务需求逐步完成，优先选择最合适的工具
- 如果需要连续调用多个工具，请一次只调用一个工具并等待结果

请清楚地向用户解释你的推理过程和操作步骤。
"""
     
async def query(query: str, mcp_server_url, model_name, base_url, api_key, temperature):
    
    client = AsyncOpenAI(
            base_url=base_url, api_key=api_key
        )

    async with sse_client(mcp_server_url) as streams:
    
        async with ClientSession(*streams) as session:

            await session.initialize()
            
            response = await session.list_tools()
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
            
            available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in response.tools]

            
            # 初始化 LLM API 调用
            response = await client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=messages,
                tools=available_tools,
                stream=True
            )
            # message = response.choices[0].message
            full_response = ""
            tool_call_text = ""
          
            while True:
                func_call_list = []
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield tool_call_text, full_response.replace('<think>', "").replace('</think>', "")  # 流式返回当前累积内容
                    elif chunk.choices[0].delta.tool_calls:
                        
                        for tcchunk in chunk.choices[0].delta.tool_calls:
                            if len(func_call_list) <= tcchunk.index:
                                func_call_list.append({
                                    "id": "",
                                    "name": "",
                                    "type": "function", 
                                    "function": { "name": "", "arguments": "" } 
                                })
                            tc = func_call_list[tcchunk.index]
                            if tcchunk.id:
                                tc["id"] += tcchunk.id
                            if tcchunk.function.name:
                                tc["function"]["name"] += tcchunk.function.name
                            if tcchunk.function.arguments:
                                tc["function"]["arguments"] += tcchunk.function.arguments
                
                        
                if not func_call_list:
                    break
                
                full_response += '🛠️ 调用工具...\n'
                yield tool_call_text, full_response.replace('<think>', "").replace('</think>', "")
                
                for tool_call in func_call_list:
                    print(tool_call)
                    tool_name = tool_call['function']['name']
                    if tool_call['function']['arguments']:
                        tool_args = json.loads(tool_call['function']['arguments'])
                    else:
                        tool_args = {}

                    # 执行工具调用
                    result = await session.call_tool(tool_name, tool_args)
                    # 记录调用详情到状态栏
                    tool_call_text += f"✅ 工具返回: {tool_name}\n参数: {tool_args}\n结果: {str(result.content)}\n---\n"
                    yield tool_call_text, full_response.replace('<think>', "").replace('</think>', "")  # 先更新状态栏
                    
                    # 将工具调用和结果添加到消息历史
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_call['id'],
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args)
                                }
                            }
                        ]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call['id'],
                        "content": str(result.content)
                    })

                # 将工具调用的结果交给 LLM
                response = await client.chat.completions.create(
                    model=model_name,
                    temperature=temperature,
                    messages=messages,
                    tools=available_tools,
                    stream=True)
            
                

with gr.Blocks() as demo:
    gr.Markdown("## MCP 客户端")
    
    # 左右分栏布局
    with gr.Row():
        # 左侧参数输入栏
        with gr.Column(scale=1):
            gr.Markdown("### 🧠 大模型配置")
            model_name = gr.Textbox(
                label="模型名称"
            )
            base_url = gr.Textbox(
                label="API 地址"
            )
            api_key = gr.Textbox(
                label="API Key",
                type="password"
            )
            temperature = gr.Number(
                label="温度",
                value=0.0,
            )
            
            gr.Markdown("### 🌐 MCP 服务配置")
            mcp_server_url = gr.Textbox(
                label="MCP 服务地址"
            )
            
            # 工具调用状态面板
            tool_status = gr.Textbox(
                label="🛠️ 工具调用记录",
                lines=10,
                interactive=False,
                autoscroll=True,
            )

        # 右侧输出区域
        with gr.Column(scale=2):
            gr.Markdown("### 💬 交互窗口")
            result_display = gr.Textbox(
                label="🧠 模型输出",
                lines=35,
                show_copy_button=True,
            )
    
    # 最底部问题输入区
    with gr.Row():
        query_input = gr.Textbox(
            label="❓ 输入你的问题",
            placeholder="输入问题后点击生成按钮...",
            scale=4
        )
        generate = gr.Button(
            "🚀 开始生成",
            scale=1,
            variant="primary"
        )
    
    generate.click(fn=query, inputs=[query_input, mcp_server_url, model_name, base_url, api_key, temperature], outputs=[tool_status, result_display])
    

    
    
    
if __name__ == "__main__":
    demo.queue().launch(server_name='0.0.0.0', allowed_paths=['./'])

