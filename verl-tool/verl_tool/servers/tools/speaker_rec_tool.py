"""
Search Retrieval Tool for verl-tool - Compatible with Search-R1 functionality
"""
from .base import BaseTool, register_tool
import regex as re
import requests
from typing import Tuple, Dict, Any, List
import logging
import json
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
lrm_path = os.path.join(current_dir, '../../../../LRM')
lrm_path = os.path.abspath(lrm_path)  # 转换为绝对路径
sys.path.insert(0, lrm_path)
from utils import get_subtitles,locate_caption,get_all_level_captions,get_relations,relation_tool,relation_error_info_translated_for_llm,output_decode
from config import SYSTEM_PROMPT,CONTEXT_LENGTH,SIMILARITY_DIFFERENCE_THREHOLD,DROPOUT_PROB,MESSAGE_MAX_LENGTH,tool_names
from data_preprocess import preprocess

logger = logging.getLogger(__name__)

@register_tool
class SearchRetrievalTool(BaseTool):

    tool_type = "speaker_rec_tool"  # 必须设置工具类型,需要和文件名一致
    
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
        if 'tool' not in parsed:
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
                if(parsed["tool"] not in tool_names):
                    info_called = f"""这是一条报错信息，你上一次的输出将被视为无效。请你根据报错内容重新进行合法的输出，且在输出中不要加入对所犯错误的陈述或反思，直接输出更改后的内容即可。报错信息具体如下：
---------------------------
你在tool标识中所给出的工具名，不存在于工具介绍列表中。请你确保给出的工具名与工具介绍中存在的工具名完全匹配。
---------------------------"""    
                elif((tool_names[parsed["tool"]] == 'non_para') and ('parameter' in parsed)):
                  
                    info_called = f"""这是一条报错信息，你上一次的输出将被视为无效。请你根据报错内容重新进行合法的输出，且在输出中不要加入对所犯错误的陈述或反思，直接输出更改后的内容即可。报错信息具体如下：
---------------------------
你在tool标识中所给出的工具名对应工具不需要参数，请勿在其后以括号形式增加参数。
---------------------------"""
                elif((tool_names[parsed["tool"]] != 'non_para') and ('parameter' not in parsed)):
                            
                            info_called = f"""这是一条报错信息，你上一次的输出将被视为无效。请你根据报错内容重新进行合法的输出，且在输出中不要加入对所犯错误的陈述或反思，直接输出更改后的内容即可。报错信息具体如下：
---------------------------
你在tool标识中所给出的工具名对应工具需要参数，请在其后以规定形式输出参数。
---------------------------"""
                elif(parsed["tool"] == "audio_similarity"):
                    candidate_score_dict = extra_field["candidate_score_dict"]
                    audio_sim_list_0=[]
                    for item1,item2 in candidate_score_dict.items():
                        if (item2 == None):
                            continue
                        audio_sim_list_0.append({"name":item1,"score":item2})
                    audio_sim_list = sorted(audio_sim_list_0, key=lambda x: x["score"], reverse=True)
                    if((extra_field["data_type"] == "second_ids") and (extra_field["revised"] == True)):
                        tt = audio_sim_list[0]["name"]
                        audio_sim_list[0]["name"] = audio_sim_list[1]["name"]
                        audio_sim_list[1]["name"] = tt  
                        score_1 = str(float(audio_sim_list[0]["score"]) - SIMILARITY_DIFFERENCE_THREHOLD/2)
                        score_0 = str(float(audio_sim_list[1]["score"]) + SIMILARITY_DIFFERENCE_THREHOLD/2) 
                        audio_sim_list[0]["score"] = score_0
                        audio_sim_list[1]["score"] = score_1 
                                
                    audio_lines = []
                    for lineitem in audio_sim_list:
                        line = f"{lineitem['name']}: {lineitem['score']}"
                        audio_lines.append(line)
                    audio_sim = "\n".join(audio_lines)
                    info_called = f"""音频相似度被整理为文本格式，每行表示一位候选人的音频相似度信息，格式为"说话人: 相似度"。例如"小明: 0.50"。其中相似度是一个0-1之间的浮点数，表示角色语音库与当前台词的语音相似度。信息中包含除“其他”选项外所有真实候选人，也可能含有场景中的未知角色（算作"其他"选项）。具体如下。
---------------------------
{audio_sim}
---------------------------"""
                elif(parsed["tool"] == "description_overall"):
                    captions = preprocess(series_name = extra_field["series_name"], option = "description")
                    s_frame = extra_field["s_frame"]
                    e_frame = extra_field["e_frame"] 
                    caption_l2 = locate_caption(captions,s_frame,e_frame,extra_field["episode_id"],7,2,6) 
                    l2_des = caption_l2["L2_info"]["L2_description"]["L2_brief_description"]
                    info_called = f"""当前场景故事背景如下:
---------------------------
{l2_des}
---------------------------"""
                elif(parsed["tool"] == "description_detailed"):
                    captions = preprocess(series_name = extra_field["series_name"], option = "description")
                    s_frame = extra_field["s_frame"]
                    e_frame = extra_field["e_frame"]  
                    caption_l1 = locate_caption(captions,s_frame,e_frame,extra_field["episode_id"],7,1,6)  
                    l1_des = caption_l1["caption"]
                    info_called = f"""当前台词画面描述细节如下：
---------------------------
{l1_des}
---------------------------"""
                elif(parsed["tool"] == "relation"):
                    relations,name_map = preprocess(series_name = extra_field["series_name"], option = "relation")
                    candidate_list = extra_field["candidate_list"]
                    all_rel_list=[]
                    for thechara in candidate_list:
                        if thechara in name_map:
                            thechara = name_map[thechara]
                        if thechara == "无效": 
                            continue
                        if(thechara not in relations["characters"]):
                            for char in relations["characters"]:
                                if ((thechara in char) or (char in thechara)):
                                    thechara = char 
                                    break
                        relation_list = relation_tool(["A",thechara],relations)
                        if(isinstance(relation_list,str)):
                            pass 
                        else:
                            all_rel_list.extend(relation_list)
                    if(len(all_rel_list)==0):  
                        info_called = "角色关系图暂时不可用。"
                    else:
                        relation_info = ""
                        for rel in all_rel_list:
                            relation_info += f"{rel[0]}是{rel[1]}的{rel[2]}。\n"
                        info_called = f"""查询到的所有人物关系如下：
---------------------------
{relation_info}
---------------------------"""
                else:
                    pass 
                    
                observation = info_called
                done = False
                valid = True
                parsed_action = parsed['tool']
                
            except Exception as e:
                logger.error(f"Require caption error for trajectory {trajectory_id}: {e}")
                observation = ""
                done = False
                valid = False  
                parsed_action = ""            

        
        self.update_env(trajectory_id, env, parsed_action, valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    