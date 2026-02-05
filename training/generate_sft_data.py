import json
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"LRM"))
from config import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument('--input_filename', type=str)
parser.add_argument('--dataset_format', type=str)  # alpaca/sharegpt
args = parser.parse_args()
output_filename = args.input_filename[:-5] + "_" + args.dataset_format + ".json"

def get_input_data():
    with open(args.input_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["sft_data"]

def convert_to_alpaca_format(data):
    alpaca_data = []
    for item in data:
        history = []
        j = item["subtitle_id"]
        candidate_list = item["candidate_list"]
        candidate_str=""
        for kk in range(len(candidate_list)):
            candidate_str += f"{kk+1}.{candidate_list[kk]}\n"
        context = item["context"]
        subtitle_lines = []
        for lineitem in context:
            line = f"[{lineitem['id']}] {lineitem['role']}: {lineitem['text']}"
            subtitle_lines.append(line)
        json_subtitles = "\n".join(subtitle_lines)
        
        system_prompt=SYSTEM_PROMPT
        
        user_prompt=f"""1.说话人候选人名单如下:
-----------------------
{candidate_str}
-----------------------
注：候选人中，“其他”表示说话人是除了以上其余候选人外的其他角色。通常来说这可能意味着说话人是影视剧中的临时角色（例如广播，警察，路人，工作人员等）或并未出现在情景现场的主要角色。
2.上下文台词被整理成文本格式，每行表示一句台词，格式为"[序号] 说话人: 台词"。例如"[1] 小明: 你好"。如果说话人为"未知"则代表此处的说话人身份暂时无法确定，为“其他”时含义同上。
台词具体内容如下：
-----------------------
{json_subtitles}
------------------------
3.你需要判断的目标台词序号为{j}。

"""
        
        input_list = [user_prompt]
        output_list = []
        if_input = False 
        for text in item["conversation"]:
            if(if_input):
                input_list.append(text)
            else:
                output_list.append(text)
            if_input = not if_input   
       
    
        for i in range(len(input_list)):
            alpaca_data.append({
                'instruction': input_list[i],
                'system': system_prompt,
                'input': "",
                'output': output_list[i],
                'history': history.copy()
            })
            history.append([input_list[i], output_list[i]])
        ##实现了每步决策形成一个数据
    
    return alpaca_data

if __name__ == "__main__":
    ori_data = get_input_data()
    if(args.dataset_format == "alpaca"):
        sft_data = convert_to_alpaca_format(ori_data)
    else:
        raise UndefinedError
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)