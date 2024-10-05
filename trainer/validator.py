import re
import subprocess

def extract_python(string_list):
    string = ' '.join(string_list)
    matches = re.findall(r'```(?:python|py)?(.*?)```', string, re.DOTALL)
    if not matches:
        matches = re.findall(r'`([^`]*)`', string)
    if matches:
        return max(matches, key=len)
    return ""

def run_script(script: str) -> int:
    try:
        result = subprocess.run(['python', '-c', script], capture_output=True, text=True, check=True)
        return -1
    except subprocess.CalledProcessError as e:
        error_line = None
        for line in e.stderr.split('\n'):
            match = re.search(r'File "<string>", line (\d+)', line)
            if match:
                error_line = int(match.group(1))
                break
        return error_line if error_line is not None else -1
    
def split_list(list, splitter):
    out_list = []
    buffer_list = []
    for token in list:
        buffer_list.append(token)
        if splitter in token:
            out_list.append(buffer_list)
            buffer_list = []
    return out_list

def error_line_to_training_dict(lineid: int, token_list: list[str], is_blank: bool = False) -> dict:
    lines = split_list(token_list, '\n')
    token_dict = {}
    for i, tokens in enumerate(lines):
        for token in tokens:
            token_dict[token] = -1 if (i-1 == lineid and lineid != -1) or is_blank else 1
    return token_dict

def is_correct(response: list[str]):
    script = extract_python(response)
    is_blank = False
    if script.strip() == "":
        is_blank = True
    error_line = run_script(script)
    out_dict = error_line_to_training_dict(error_line, response, is_blank)
    return out_dict
