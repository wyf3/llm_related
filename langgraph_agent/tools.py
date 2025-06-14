from langchain_core.tools import tool
import os
import traceback
import subprocess
@tool
def create_file(file_name, file_contents):
    """
    Create a new file with the provided contents at a given path in the workspace.
    
    args:
        file_name (str): Name to the file to be created
        file_contents (str): The content to write to the file
    """
    try:

        file_path = os.path.join(os.getcwd(), file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            file.write(file_contents)

        return {
            "message": f"Successfully created file at {file_path}"
        }

    except Exception as e:
        return {
            "error": str(e)
        }

@tool
def str_replace(file_name, old_str, new_str):
    """
    Replace specific text in a file.
    
    args:
        file_name (str): Name to the target file
        old_str (str): Text to be replaced (must appear exactly once)
        new_str (str): Replacement text
    """
    try:
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, "r") as file:
            content = file.read()

        new_content = content.replace(old_str, new_str, 1)
        
        with open(file_path, "w") as file:
            file.write(new_content)

        return {"message": "Successfully replaced '{old_str}' with '{new_str}' in {file_path}"}
    except Exception as e:
        return {"error": f"Error replacing '{old_str}' with '{new_str}' in {file_path}: {str(e)}"}

@tool
def send_message(message: str):
    """
    send a message to the user
    
    args:
        message: the message to send to the user
    """
    
    return message

@tool
def shell_exec(command: str) -> dict:
    """
    在指定的 shell 会话中执行命令。

    参数:
        command (str): 要执行的 shell 命令

    返回:
        dict: 包含以下字段：
            - stdout: 命令的标准输出
            - stderr: 命令的标准错误
    """
  
    try:
        # 执行命令
        result = subprocess.run(
            command,
            shell=True,          
            cwd=os.getcwd(),        
            capture_output=True,
            text=True,    
            check=False
        )

        # 返回结果
        return {"message":{"stdout": result.stdout,"stderr": result.stderr}}

    except Exception as e:
        return {"error":{"stderr": str(e)}}