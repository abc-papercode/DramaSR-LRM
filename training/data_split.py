import json
import argparse 
from datetime import datetime
import random
import math

SIMILARITY_DIFFERENCE_THREHOLD = 0.03

parser = argparse.ArgumentParser()
parser.add_argument('--series_name',type=str)
parser.add_argument('--end_name',type=str)
parser.add_argument('--if_valid',type=str) ##是否要额外保留验证集
parser.add_argument('--train_ratio',type=float,default=0.9) ##当额外保留验证集时，用于训练的比例
parser.add_argument('--epi_cnt',type=int) ##剧集数量
parser.add_argument('--sft_cnt',type=int) ##训练数据中用于sft的剧集数量
parser.add_argument('--second_revising_ratio',type=float,default=0.5) ##rl中，second类中多少比例进行扰动
parser.add_argument('--first_ratio',type=float,default=0.2) ##最终数据中 first类占比

args = parser.parse_args()



id_file= f"../LRM/test_result/statistic_count/tool_master_{args.series_name}_important_ids_{args.end_name}.json" #here
data_file= f"../LRM/test_result/statistic_count/tool_master_{args.series_name}_all_data_{args.end_name}.json" #here


timestamp = datetime.now().date()
output_split_data_file = f"tool_master_{args.series_name}_split_data_{timestamp}.json" ##数据划分文件（包含数据本体） #here
output_meta_data_file = f"tool_master_{args.series_name}_meta_data_{timestamp}.json" ##数据划分元文件（包含数据划分信息） #here


split_meta_data={} 
training_data = []
training_set = {}


def get_input_data():
    with open(id_file, 'r', encoding='utf-8') as f:
        important_ids = json.load(f)
    with open(data_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    return important_ids,all_data


if __name__ == "__main__":
    important_ids,all_data = get_input_data()
    epi_cnt = args.epi_cnt
    second_data_over_thres=[]
    second_data_below_thres=[]


    if(args.if_valid == "no"):
        valid_epi_cnt = 0
        split_meta_data["valid_epi_cnt"] = 0
        valid_epis = []
    else: 
        valid_epi_cnt = int(epi_cnt * (1-args.train_ratio)) ## 用作测试的集数
        if(valid_epi_cnt == 0):
            valid_epi_cnt = 1 
        valid_epis = sorted(random.sample(range(1, epi_cnt + 1), valid_epi_cnt)) ##用作测试的集id
        split_meta_data["valid_epi_cnt"] = valid_epi_cnt
    split_meta_data["valid_epis"] = valid_epis

    rl_cnt = epi_cnt - valid_epi_cnt - args.sft_cnt
    sft_cnt = args.sft_cnt

    sft_epis=[]
    rl_epis=[]
    sft_accum = 0
    for ii in range(epi_cnt):
        epi_id = ii+1
        if(epi_id in valid_epis):
            continue
        if(sft_accum < sft_cnt):
            sft_epis.append(epi_id)
            sft_accum += 1
        else:
           rl_epis.append(epi_id) 

    split_meta_data["sft_cnt"] = sft_cnt
    split_meta_data["sft_epis"] = sft_epis
    split_meta_data["rl_cnt"] = rl_cnt
    split_meta_data["rl_epis"] = rl_epis

    ##先处理rl：  
    rl_data=[]
    rl_cou = 0
    for epid in rl_epis:   
        for case in all_data[str(epid)].values():
            if (case["data_type"] == "second_ids") and (case["dif_sim"] <= SIMILARITY_DIFFERENCE_THREHOLD): #按照比例做数据扰动
                if(random.random() < args.second_revising_ratio):
                    tt = case["candidate_list"][0]
                    case["candidate_list"][0] = case["candidate_list"][1]
                    case["candidate_list"][1] = tt  ##音频相似度部分会在verl的工具函数设定中进行扰动
                    case["revised"] = True
            rl_data.append(case)
            rl_cou += 1
            
    random.shuffle(rl_data)
    training_set["rl_data"] = rl_data
    split_meta_data["rl_cases"] = rl_cou
    ##rl结束：

    
    ##再处理sft：
    for item in important_ids["first_ids"]:
        episode = item[0]
        subtitle = item[1]
        if(episode not in sft_epis):
            continue ## 不放在训练集
        training_data.append(all_data[str(episode)][str(subtitle)]) 
    split_meta_data["sft_first_ids_cnt"] = len(training_data)
        
    for item in important_ids["second_ids"]:
        episode = item[0]
        subtitle = item[1]
        if(episode not in sft_epis):
            continue ## 不放在训练集
        ori_data = all_data[str(episode)][str(subtitle)]
        if ori_data["dif_sim"] > SIMILARITY_DIFFERENCE_THREHOLD:
            second_data_over_thres.append(item)
        else:
            training_data.append(ori_data) 
    split_meta_data["sft_second_data_below_thres"] = len(training_data) - split_meta_data["sft_first_ids_cnt"]

           
           
    first_cnt = split_meta_data["sft_first_ids_cnt"]
    second_cnt = math.ceil(first_cnt*(1-args.first_ratio)/args.first_ratio) 
    remaining_cnt = second_cnt - split_meta_data["sft_second_data_below_thres"]
    if(remaining_cnt <= 0):
        remaining_cnt = 0
    random.shuffle(second_data_over_thres)##随机顺序
    second_final_data = second_data_over_thres[:remaining_cnt]
    for item in second_final_data:
        episode = item[0]
        subtitle = item[1]
        training_data.append(all_data[str(episode)][str(subtitle)]) 
    split_meta_data["sft_second_data_bover_thres"] = remaining_cnt
           
    random.shuffle(training_data)
    training_set["sft_data"] = training_data
    
    split_meta_data["series_name"] = args.series_name
    
    with open(output_split_data_file, 'w', encoding='utf-8') as f:
        json.dump(training_set, f, ensure_ascii=False, indent=2)
    with open(output_meta_data_file, 'w', encoding='utf-8') as f:
        json.dump(split_meta_data, f, ensure_ascii=False, indent=2)
            
    