from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Optional
from openai import AsyncOpenAI
from contextlib import AsyncExitStack
import json
import asyncio
import os
from prompts import *
from search_mcp import logger


base_url = "https://openrouter.ai/api/v1"
api_key = 'aaa'
model_name = 'deepseek/deepseek-chat:free'
def get_clear_json(text):
    if '```json' not in text:
        return 0, text
    
    return 1, text.split('```json')[1].split('```')[0]

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        
    async def connect_to_server(self, server_script_path: str):
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # 列出可用工具
        response = await self.session.list_tools()
        tools = response.tools
        logger.info(f"\nConnected to server with tools: {[tool.name for tool in tools]}")
    
    
    async def process_query(self, query: str) -> str:
        """使用 LLM 和 MCP 服务器提供的工具处理查询"""
        
        response = await self.session.list_tools()
        
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]
        logger.info(f'available_tools:\n\n{available_tools}')
        
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + str(available_tools)
            },
            {
                "role": "user",
                "content": query
            }
        ]
        response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages
            )
        
        message = response.choices[0].message
        logger.info(f'llm_output(tool call)：{message.content}')
        
        results = []
        while True:
            
            flag, json_text = get_clear_json(message.content)
            
            if flag == 0:
                response = await self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": query}]
            )
                return response.choices[0].message.content
    
            json_text = json.loads(json_text)
            tool_name = json_text['name']
            tool_args = json_text['params']
            result = await self.session.call_tool(tool_name, tool_args)
            logger.info(f'tool name: \n{tool_name}\ntool call result: \n{result}')
            results.append(result.content[0].text)
            
            messages.append({
                "role": "assistant",
                "content": message.content
            })
            messages.append({
                "role": "user",
                "content": f'工具调用结果如下：{result}'
            })
            
            messages.append({
                "role": "user",
                "content": NEXT_STEP_PROMPT.format(query)
            })
            
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            
            message = response.choices[0].message
            logger.info(f'llm_output：\n{message.content}')
            
            if 'finish' in message.content:
                break
            
            messages.append({
                "role": "assistant",
                "content": message.content
            })
        
        
        
        
        messages.append({
                "role": "user",
                "content": FINISH_GENETATE.format('\n\n'.join(results), query)
                })
        
        response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages
            )
        
        message = response.choices[0].message.content
        return message
    
    async def chat_loop(self):
        """运行交互式聊天循环"""
        logger.info("\nMCP Client Started!")
        logger.info("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                print(response)
            except Exception as e:
                logger.error(f"\nError: {str(e)}")
                


async def main():
   
    client = MCPClient()
    
    
    await client.connect_to_server('./search_mcp.py')
    
    await client.chat_loop()
    
    

if __name__ == "__main__":
    asyncio.run(main())