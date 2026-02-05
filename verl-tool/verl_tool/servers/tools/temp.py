"""
Search Retrieval Tool for verl-tool - Compatible with Search-R1 functionality
"""
from .base import BaseTool, register_tool
import regex as re
import requests
from typing import Tuple, Dict, Any, List
import logging
from .lyx_utils import output_decode,get_subtitles,get_captions,locate_caption,get_all_level_captions,get_relations,relation_tool,relation_error_info_translated_for_llm

logger = logging.getLogger(__name__)

data_path = "/opt/huawei/explorer-env/algorithm/foraudio/vllm_speaker/training/vpr_dataset_lyx.json" ##按集分割后的非常初始的标签传播数据
subtitle_dir = "/opt/huawei/explorer-env/dataset/foraudio/vllm_speaker/vllm/subtitle_data/"
caption_dir = "/opt/huawei/explorer-env/dataset/foraudio/vllm_speaker/vllm/captions_crop/"
relation_dir = "/opt/huawei/explorer-env/dataset/foraudio/vllm_speaker/vllm/relation_data/"

CONTEXT_LENGTH = 30
SIMILARITY_DIFFERENCE_THREHOLD = 0.03

@register_tool
class SearchRetrievalTool(BaseTool):
    
    def __init__(self):
        super().__init__()
        self.tool_type = "speaker_rec"  # 必须设置工具类型

    def extract_and_validate(self, action):
        output = action
        pattern = r'<(think|tool|answer)>(.*?)</\1>'
    
        if not isinstance(output, str):
            return {"error": "你的输出内容不为字符串格式。"}
        # 使用正则表达式查找所有匹配项
        matches = re.finditer(pattern, output,re.DOTALL)
    
        # 如果没有找到任何匹配项，返回错误信息
        if not matches:
            return {"error": "未找到标识符对。"}
    
        # 创建一个字典来存储提取的信息
        result = {}
    
    
        for match in matches:
            key, value = match.groups()
            start, end = match.span()
        
            # 如果key已经存在于字典中，返回错误信息
            if key in result:
                return {"error": f"{key}标识符对多次出现。"}
        
            result[key] = value
            last_end = end
    
        # 标识符出现情况要求
    
        if 'think' not in result:
            return {"error": "think标识符对未出现。"}
    
        if ('tool' in result) == ('answer' in result):
            return {"error": "未满足tool与answer标识符对中出现且仅出现一对的条件。"}
    
        ##此时整体格式检查完毕，接下来检查tool的格式
    
        if 'tool' in result:
            tool_value = result['tool']
            # 使用正则表达式匹配工具名和参数
            tool_pattern = r'(\w+)\((.*?)\)'
            tool_match = re.match(tool_pattern, tool_value)
        
            if tool_match: ##工具名中有括号，即带参数
                tool_name, parameters = tool_match.groups()
                # 将参数按逗号分割并去除空格
                parameter_list = [param.strip() for param in parameters.split(',')]
            
                # 更新result字典
                result['tool'] = tool_name
                result['parameter'] = parameter_list
            ##否则认为工具名不带参数，字符串为工具名本身
        
        ##此时如果格式无误，parameter中会存储参数list，tool会存储工具名
    
        # 返回JSON格式的结果
        return result
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute search query via retrieval service.
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string containing search query
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed = output_decode(action)
        env = self.load_env(trajectory_id)
        if 'tool' not in  parsed:
            if 'answer' in parsed:
                observation = ""
                done = True
                valid = False
                parsed_action = parsed['answer']
            else:
                observation = ""
                done = False
                valid = False 
                parsed_action = ""
        else:
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    audio_data = json.load(f)
                episode_for_input = audio_data[extra_field["episode_id"]-1]
                subtitles = get_subtitles(extra_field["episode_id"],subtitle_dir)
                if(parsed["tool"] == "audio_similarity"):
                    candidate_score_dict = episode_for_input[extra_field["subtitle_id"]]["candidate_list"]
                    candidate_score_list = [value for key,value in candidate_score_dict.items()]
                    audio_sim_raw_list = candidate_score_list[:-1] ##去除其它    
                    audio_sim_list = [{key: value for key, value in item.items() if key != "idx"}for item in audio_sim_raw_list]##不要元素的idx键值
                    if((extra_field["data_type"] == "second_ids") and (extra_field["revised_result"] == True)):
                            tt = audio_sim_list[0]["name"]
                            audio_sim_list[0]["name"] = audio_sim_list[1]["name"]
                            audio_sim_list[1]["name"] = tt   ##交换名字
                            score_1 = audio_sim_list[0]["score"] - SIMILARITY_DIFFERENCE_THREHOLD/2
                            score_0 = audio_sim_list[1]["score"] + SIMILARITY_DIFFERENCE_THREHOLD/2 ##映射法；且能确保两者顺序调转
                            audio_sim_list[0]["score"] = score_0
                            audio_sim_list[1]["score"] = score_1
                            
                    audio_sim = json.dumps(audio_sim_list, ensure_ascii=False, indent=2)
                        
                    info_called = f"""音频相似度被整理为列表形式，其中每个元素是一个字典，代表一个候选人。字典包含两个键name和score。其中name键值是一个字符串，表示候选人姓名；score键值是一个浮点数，表示其语音库与当前台词的语音相似度。列表中出现的候选人是按照score表示的语音相似度由大到小排序的，包含除“其它”外所有已给出的候选人。音频相似度列表如下。
---------------------------
{audio_sim}
---------------------------"""
                elif(parsed["tool"] == "description_detailed"):
                    captions = get_all_level_captions(caption_dir)
                    lower_bound = max(0,extra_field["subtitle_id"]-CONTEXT_LENGTH)
                    upper_bound = min(len(episode_for_input)-1,extra_field["subtitle_id"]+CONTEXT_LENGTH) ##上下文端点对应的数组下标
                    s_frame = subtitles[str(extra_field["subtitle_id"]+1)]["start_frame"]
                    e_frame = subtitles[str(extra_field["subtitle_id"]+1)]["end_frame"] ##下标＋1以适配subtitle文件格式
                    caption_l2 = locate_caption(captions,s_frame,e_frame,extra_field["episode_id"],7,2)  ##i+1表示第几集
                    l2_des = caption_l2["L2_info"]["L2_description"]["L2_detailed_description"] ##暂时用详细版本
                    info_called = f"""当前台词画面描述细节如下：
---------------------------
{l2_des}
---------------------------"""
                elif(parsed["tool"] == "description_overall"):
                    captions = get_all_level_captions(caption_dir)
                    lower_bound = max(0,extra_field["subtitle_id"]-CONTEXT_LENGTH)
                    upper_bound = min(len(episode_for_input)-1,extra_field["subtitle_id"]+CONTEXT_LENGTH) ##上下文端点对应的数组下标
                    s_frame = subtitles[str(extra_field["subtitle_id"]+1)]["start_frame"]
                    e_frame = subtitles[str(extra_field["subtitle_id"]+1)]["end_frame"] ##下标＋1以适配subtitle文件格式
                    caption_l1 = locate_caption(captions,s_frame,e_frame,i+1,7,1)  ##i+1表示第几集
                    l1_des = caption_l1["caption"]
                    info_called = f"""当前场景故事背景如下：
---------------------------
{l1_des}
---------------------------"""
                elif(parsed["tool"] == "relation"):
                    relations = get_relations(relation_dir)
                    relation_list = relation_tool(parsed["parameter"],relations)
                    if(isinstance(relation_list,str)):
                        error_msg = relation_error_info_translated_for_llm(relation_list) ##转化成给llm的报错信息
                        info_called = f"""这是一条报错信息，你上一次的输出将被视为无效。请你根据报错内容重新进行合法的输出。报错信息具体如下：
---------------------------
{error_msg}
---------------------------"""##直接给？
                    elif(len(relation_list)==0):
                        info_called = "角色关系图中没有符合条件的关系三元组。"
                    else:
                        relation_info = ""
                        for rel in relation_list:
                            relation_info += f"{str(rel)}\n"
                        info_called = f"""
                        查询到的所有关系三元组如下：
---------------------------
{relation_info}
---------------------------"""
                elif(parsed["tool"] == "relationship_info"):
                    relations = get_relations(relation_dir)
                    relationship_type = list(relations["relationship_type"].keys())
                    relationship_type_info = str(relationship_type)
                    characters = relations["characters"]
                    characters_info = str(characters)
                    info_called = f"""
                    所有角色名被整理成列表形式。查询到的所有角色名如下：
---------------------------
{characters_info} 
---------------------------
                    所有单向关系名被整理成列表形式，其中元素为关系名称且按照在角色关系图中出现的次数降序排列。查询到的所有单向关系名如下：
---------------------------
{relationship_type_info}
---------------------------"""
                else:
                    pass ##不会到此
                
                observation = info_called
                done = False
                valid = True
                
            except Exception as e:
                logger.error(f"Require caption error for trajectory {trajectory_id}: {e}")
                observation = ""
                done = False
                valid = False  
                parsed_action = ""            

        
        self.update_env(trajectory_id, env, parsed_action, valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    