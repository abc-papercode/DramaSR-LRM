"""
Search Retrieval Tool for verl-tool - Compatible with Search-R1 functionality
"""
from .base import BaseTool, register_tool
import regex as re
import requests
from typing import Tuple, Dict, Any, List
import logging
from .utils.video_get_caption import get_tool_observation
logger = logging.getLogger(__name__)

@register_tool
class SearchRetrievalTool(BaseTool):
    tool_type = "get_caption"
    
    def extract_and_validate(self, action):
        """
        提取 <think>、<answer>、<tool> 标签内容，并验证是否符合4种合法格式之一。
        
        参数：
            text (str): 输入字符串
        返回：
            dict: 若格式合法，返回各部分内容的字典；否则返回 {'error': '说明'}
        """
        tags = ['think', 'answer', 'tool']
        results = {}

        # 提取每个标签的内容（只取第一个）
        for tag in tags:
            pattern = fr'<{tag}>(.*?)</{tag}>'
            match = re.search(pattern, action, re.DOTALL)
            if match:
                results[tag] = match.group(1).strip()

        return results

    
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
        parsed = self.extract_and_validate(action)
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
                parsed_action = parsed['tool']
                results = get_tool_observation(parsed['tool'], extra_field['video_uid'], extra_field['data_source'], extra_field['width'], extra_field['fps'])
                if results is None:
                    observation = ""
                    done = False
                    valid = False

                else:
                    observation = results
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
    