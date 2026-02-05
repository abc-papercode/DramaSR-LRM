import json
import sys
import os 
import argparse
import random
import math
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"LRM"))
from config import SYSTEM_PROMPT
from data_preprocess import preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--input_filename', type=str)
parser.add_argument('--series_name', type=str)
parser.add_argument('--val_ratio', type=float)
parser.add_argument('--dataset_format', type=str)  # verl
args = parser.parse_args()
output_train_filename = args.input_filename[:-5] + "_" + args.dataset_format+ "_" + str(args.val_ratio) +"_train" + ".parquet"
output_val_filename = args.input_filename[:-5] + "_" + args.dataset_format+ "_" + str(args.val_ratio) +"_val" + ".parquet"

def get_input_data():
    with open(args.input_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    meta = {"series_name":args.series_name}
    return data["rl_data"],meta

def convert_to_verl_format(data,meta,speaker_data):
    verl_data = []
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
        gt_role = item["role"]
        if item["role"] not in item["candidate_list"]:
            gt_role = "其他"
        nowdata = {
                "data_source": "speaker_rec",
                "prompt": [
                    {
                    "role": "system",
                    "content": system_prompt
                    },
                    {
                    "role": "user",
                    "content": user_prompt
                    }
                ],
                "ability": "commonsense",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": gt_role
                },
                "extra_info": {
                    'episode_id': item['episode_id'],
                    'subtitle_id': item['subtitle_id'],
                    "data_type": item['data_type'],
                    "candidate_list": item["candidate_list"],
                    "revised": False,
                    "candidate_score_dict":speaker_data[item['episode_id']-1][item['subtitle_id']]["rel_info"],
                    "s_frame":speaker_data[item["episode_id"]-1][item["subtitle_id"]]["start_frame"],
                    "e_frame":speaker_data[item["episode_id"]-1][item["subtitle_id"]]["end_frame"], 
                    "series_name":meta["series_name"]
                }
            }
        if "revised" in item:
            nowdata["extra_info"]["revised"] = True  
        verl_data.append(nowdata)
    
    return verl_data

if __name__ == "__main__":
    ori_data,meta = get_input_data()
    speaker_data,episode_cnt,relations,name_map,captions = preprocess(meta["series_name"]) 
    if(args.dataset_format == "verl"):
        rl_data = convert_to_verl_format(ori_data,meta,speaker_data)
    else:
        raise UndefinedError
    
    val_cnt = math.ceil(len(rl_data)*args.val_ratio) 
    random.shuffle(rl_data)
    val_list = rl_data[:val_cnt]
    for i in range(len(val_list)):
        val_list[i]["extra_info"]["split"] = "valid" 
    train_list = rl_data[val_cnt:]
    for i in range(len(train_list)):
        train_list[i]["extra_info"]["split"] = "train"
        
    df_val = pd.DataFrame(val_list)
    df_val.to_parquet(output_val_filename, index=False)
    df_train = pd.DataFrame(train_list)
    df_train.to_parquet(output_train_filename, index=False)
    