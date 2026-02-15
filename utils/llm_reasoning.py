import torch
import torch.nn as nn
import numpy as np
import requests
import json
import re
import logging
import time
import os
from typing import Dict, Any, Tuple
from config import cfg
# 注意: 需要安装 openai 库 (pip install openai)
from openai import OpenAI

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# 1. 语义序列化器 -
# 实现异构特征到结构化自然语言的转换，为LLM推理提供完整上下文
# ==========================================
class SemanticSerializer:
    @staticmethod
    def _analyze_ego_kinematics(hist_traj: np.ndarray) -> Dict[str, Any]:
        if hist_traj.shape[0] == 0:
            return {"desc": "Unknown", "v_kph": 0, "acc_avg": 0, "yaw_rate": 0}
        # 当前帧状态
        curr_state = hist_traj[-1]
        # 速度分析
        v_mps = np.linalg.norm(curr_state[2:4])
        v_kph = v_mps * 3.6
        # 加速度趋势分析（1s窗口）
        window = min(10, len(hist_traj) - 1)
        v_prev = np.linalg.norm(hist_traj[-window, 2:4])
        acc_avg = (v_mps - v_prev) / (0.1 * window)
        # 转向率分析
        yaw_curr = curr_state[6]
        yaw_prev = hist_traj[-window, 6]
        yaw_diff = yaw_curr - yaw_prev
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi  # 归一化到[-π, π]
        yaw_rate = yaw_diff / (0.1 * window)

        # 语义映射
        if v_kph < 1.0:
            speed_desc = "Stationary (speed < 1 km/h)"
        elif v_kph < 20:
            speed_desc = f"Low-speed driving ({v_kph:.1f} km/h)"
        elif v_kph < 60:
            speed_desc = f"Urban normal driving ({v_kph:.1f} km/h)"
        else:
            speed_desc = f"High-speed driving ({v_kph:.1f} km/h)"

        if acc_avg > 0.8:
            acc_desc = "accelerating"
        elif acc_avg < -3.5:
            acc_desc = "emergency braking"
        elif acc_avg < -0.8:
            acc_desc = "decelerating"
        else:
            acc_desc = "maintaining constant speed"

        TURN_THRESH = 0.05
        if yaw_rate > TURN_THRESH:
            turn_desc = "turning left"
        elif yaw_rate < -TURN_THRESH:
            turn_desc = "turning right"
        else:
            turn_desc = "going straight"

        return {
            "full_desc": f"Ego vehicle is {speed_desc}, {acc_desc}, {turn_desc}.",
            "v_kph": v_kph,
            "acc_avg": acc_avg,
            "yaw_rate": yaw_rate,
            "acc_desc": acc_desc,
            "turn_desc": turn_desc
        }

    @staticmethod
    def _analyze_surrounding_agents(surrounding_agents: list) -> str:
        """周围智能体交互分析"""
        if not surrounding_agents:
            return "No surrounding vehicles in the scene."
        agent_descs = []
        for agent in surrounding_agents:
            rel_dist = agent['rel_dist']
            rel_dir = agent['rel_dir']
            v_kph = agent['v_kph']
            action = agent['action']
            agent_descs.append(
                f"- A vehicle is {rel_dist:.1f}m {rel_dir} of ego, driving at {v_kph:.1f} km/h, {action}."
            )
        return "Surrounding vehicles:\n" + "\n".join(agent_descs)

    @staticmethod
    def _analyze_map_scene(map_feat: np.ndarray) -> Dict[str, Any]:
        """地图场景语义分析"""
        if map_feat is None:
            return {"desc": "Map information unavailable.", "has_intersection": False, "curve_dir": "straight"}
        if isinstance(map_feat, torch.Tensor):
            map_data = map_feat.cpu().numpy()
        else:
            map_data = map_feat

        # 筛选自车前方有效车道
        line_centers = map_data[..., 0:2].mean(axis=1)
        valid_mask = (line_centers[:, 0] > 5) & (line_centers[:, 0] < 50) & (np.abs(line_centers[:, 1]) < 15)
        if not np.any(valid_mask):
            return {"desc": "Open road ahead, no lane constraints.", "has_intersection": False, "curve_dir": "straight"}

        valid_lines = map_data[valid_mask]

        if len(valid_lines) == 0:
            return {"desc": "Open road ahead, no clear lanes.", "has_intersection": False, "curve_dir": "straight"}

        line_starts = valid_lines[:, 0, 0:2]
        line_ends = valid_lines[:, -1, 0:2]
        line_vecs = line_ends - line_starts
        angles = np.arctan2(line_vecs[:, 1], line_vecs[:, 0])

        # 路口判定
        has_left = np.any(angles > 0.3)
        has_right = np.any(angles < -0.3)
        has_intersection = has_left and has_right

        # 弯道判定
        avg_angle = np.mean(angles)
        if avg_angle > 0.15:
            curve_dir = "left"
            map_desc = "The lane curves to the left ahead."
        elif avg_angle < -0.15:
            curve_dir = "right"
            map_desc = "The lane curves to the right ahead."
        else:
            curve_dir = "straight"
            map_desc = "Straight road ahead."

        if has_intersection:
            map_desc = "Approaching an intersection with multiple branching lanes ahead."

        # 交通灯状态
        if valid_lines.shape[-1] > 6:
            light_status = valid_lines[..., 6].mean()
            if light_status == 0:
                light_desc = "Traffic light is red, must stop before the stop line."
            elif light_status == 1:
                light_desc = "Traffic light is yellow, prepare to stop."
            else:
                light_desc = "Traffic light is green, can pass normally."
        else:
            light_status = -1
            light_desc = "Traffic light status unknown."

        return {
            "full_desc": f"{map_desc} {light_desc}",
            "has_intersection": has_intersection,
            "curve_dir": curve_dir,
            "light_status": light_status
        }

    @staticmethod
    def _analyze_risk_level(meta: Dict, kinematics: Dict) -> str:
        """风险等级分析"""
        is_accident_prone = meta.get('is_accident', False)
        is_emergency = "emergency braking" in kinematics['acc_desc']
        if is_emergency:
            return "High risk: Emergency braking detected, immediate collision avoidance required."
        if is_accident_prone:
            return "Medium risk: Accident-prone scenario, need to drive defensively."
        return "Low risk: Normal driving scenario, maintain safe driving."

    @staticmethod
    def generate_cot_prompt(raw_hist: np.ndarray, map_feat: np.ndarray, surrounding_agents: list, meta: Dict) -> str:
        """三阶CoT推理Prompt"""
        # 1. 解析各维度语义
        kine_data = SemanticSerializer._analyze_ego_kinematics(raw_hist)
        map_data = SemanticSerializer._analyze_map_scene(map_feat)
        agent_desc = SemanticSerializer._analyze_surrounding_agents(surrounding_agents)
        risk_desc = SemanticSerializer._analyze_risk_level(meta, kine_data)

        # 2. 动作空间描述
        action_desc = [f"- ID {aid}: {name}" for aid, name in cfg.ACTION_SPACE.items()]
        action_str = "\n".join(action_desc)

        # 3. 严格遵循三阶CoT逻辑的Prompt模板
        prompt = f"""
[SYSTEM ROLE]
You are the safety decision-making core of an autonomous driving vehicle. Your task is to perform Chain-of-Thought (CoT) reasoning strictly following the three-stage logic below, and output the final meta-action ID. You must prioritize traffic safety and compliance with traffic rules in all decisions.

[THREE-STAGE COT REASONING RULES - MUST FOLLOW]
Stage 1: Risk Perception. Identify all key risk sources in the scene, including ego vehicle state, surrounding vehicles' behavior, map constraints, and traffic light status.
Stage 2: Rule Association. Invoke traffic rules and defensive driving knowledge based on the perceived risks. The priority of rules is: Physical Safety > Traffic Regulations > Driving Efficiency.
Stage 3: Intention Locking. Based on the above reasoning, select the only most appropriate meta-action from the action space, and output its integer ID.

[PERCEPTION INPUTS]
1. Ego Vehicle State: {kine_data['full_desc']}
2. Map & Traffic Rules: {map_data['full_desc']}
3. Surrounding Vehicles Interaction: {agent_desc}
4. Scene Risk Level: {risk_desc}

[ACTION SPACE]
{action_str}

[OUTPUT REQUIREMENTS]
You must output a valid JSON object only, without any other text. The JSON format is strictly as follows:
{{
    "stage1_risk_perception": "Detailed description of all risk sources identified",
    "stage2_rule_association": "Detailed description of the traffic rules and defensive driving logic invoked",
    "stage3_intention_locking": "Brief explanation of the final action selection",
    "action_id": <integer ID of the selected meta-action>
}}
"""
        return prompt.strip()


# ==========================================
# 2. 规则降级模块 - 安全冗余
# 当LLM API调用失败时，自动降级为基于规则的决策，保证系统鲁棒性
# ==========================================
class RuleBasedFallback:
    @staticmethod
    def infer(prompt: str) -> Tuple[int, str]:
        prompt_lower = prompt.lower()
        # 解析关键状态
        is_stationary = "stationary" in prompt_lower
        is_emergency = "emergency braking" in prompt_lower
        is_accelerating = "accelerating" in prompt_lower
        is_decelerating = "decelerating" in prompt_lower
        # 转向与地图约束
        is_turning_left = "turning left" in prompt_lower or "curves to the left" in prompt_lower
        is_turning_right = "turning right" in prompt_lower or "curves to the right" in prompt_lower
        # 交通灯与风险
        red_light = "red" in prompt_lower and "traffic light" in prompt_lower
        high_risk = "high risk" in prompt_lower

        # 1. 最高优先级：红灯/紧急制动
        if red_light or is_emergency:
            return 2, "Fallback: Red light or emergency braking -> Straight_Dec"
        # 2. 静止状态
        if is_stationary:
            return 9, "Fallback: Stationary -> Stop"
        # 3. 转向约束（地图优先级高于运动学）
        if is_turning_left:
            if is_decelerating or high_risk:
                return 4, "Fallback: Left turn + deceleration -> Left_Dec"
            if is_accelerating:
                return 5, "Fallback: Left turn + acceleration -> Left_Acc"
            return 3, "Fallback: Left turn -> Left_Keep"
        if is_turning_right:
            if is_decelerating or high_risk:
                return 7, "Fallback: Right turn + deceleration -> Right_Dec"
            if is_accelerating:
                return 8, "Fallback: Right turn + acceleration -> Right_Acc"
            return 6, "Fallback: Right turn -> Right_Keep"
        # 4. 直行逻辑
        if is_decelerating or high_risk:
            return 2, "Fallback: Straight + deceleration -> Straight_Dec"
        if is_accelerating:
            return 1, "Fallback: Straight + acceleration -> Straight_Acc"
        # 5. 默认保持
        return 0, "Fallback: Stable driving -> Straight_Keep"


# ==========================================
# 3. LLM客户端 - 支持API/开源模型本地部署
# ==========================================
class LLMClient:
    def __init__(self, provider="deepseek"):
        self.provider = provider
        self.client = None
        self.config = self._init_config()

    def _init_config(self):
        configs = {
            "openai": {
                "api_key": cfg.OPENAI_API_KEY,
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4-turbo"
            },
            "deepseek": {
                "api_key": cfg.DEEPSEEK_API_KEY,
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat"
            },
            "local": {
                "api_key": "EMPTY",
                "base_url": "http://localhost:8000/v1",
                "model": "Qwen-2.5-72B-Instruct"
            }
        }
        selected = configs.get(self.provider, configs["deepseek"])
        if selected["api_key"]:
            self.client = OpenAI(
                api_key=selected["api_key"],
                base_url=selected["base_url"]
            )
        return selected

    def _call_api(self, prompt: str) -> str:
        if not self.client:
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.LLM_TEMPERATURE,
                max_tokens=cfg.LLM_MAX_TOKENS,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"LLM API调用失败: {e}")
            return None

    def _parse_output(self, content: str) -> Tuple[int, str]:
        if not content:
            return 0, "Empty response"
        # 清理markdown格式
        content_clean = re.sub(r'```json\s*|```', '', content, flags=re.IGNORECASE).strip()
        try:
            result = json.loads(content_clean)
            action_id = int(result.get("action_id", 0))
            full_reasoning = json.dumps(result, ensure_ascii=False, indent=2)
            # 校验动作ID合法性
            if 0 <= action_id < cfg.NUM_ACTIONS:
                return action_id, full_reasoning
            else:
                logger.warning(f"非法动作ID: {action_id}，降级到规则决策")
                return 0, "Invalid action ID"
        except Exception as e:
            logger.warning(f"LLM输出解析失败: {e}")
            # 正则兜底提取action_id
            match = re.search(r'"action_id"\s*:\s*(\d+)', content_clean)
            if match:
                action_id = int(match.group(1))
                if 0 <= action_id < cfg.NUM_ACTIONS:
                    return action_id, "Regex extracted action ID"
            return 0, "Parse failed"

    def query(self, prompt: str) -> Tuple[int, str]:
        """执行推理，自动降级保障鲁棒性"""
        if self.provider == "rule":
            return RuleBasedFallback.infer(prompt)
        # 尝试API调用
        content = self._call_api(prompt)
        if content:
            action_id, reasoning = self._parse_output(content)
            if 0 <= action_id < cfg.NUM_ACTIONS:
                return action_id, reasoning
        # API失败/解析失败，自动降级
        logger.warning("LLM推理失败，自动降级到规则决策")
        return RuleBasedFallback.infer(prompt)


# ==========================================
# 4. 元动作嵌入模块 - 对齐论文4.3.3节
# ==========================================
class ActionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(cfg.NUM_ACTIONS, cfg.HIDDEN_DIM)
        # 正交初始化，保证动作空间正交性
        nn.init.orthogonal_(self.embedding.weight)

    def forward(self, action_ids):
        return self.embedding(action_ids)


# 测试模块
if __name__ == "__main__":
    # 测试Prompt生成
    test_hist = np.zeros((cfg.OBS_LEN, cfg.INPUT_DIM))
    test_hist[:, 2] = 10.0  # 10m/s匀速
    test_map = np.zeros((cfg.MAP_MAX_LINES, cfg.MAP_POINTS_PER_LINE, cfg.MAP_DIM))
    test_meta = {'is_accident': False}
    test_agents = []

    serializer = SemanticSerializer()
    prompt = serializer.generate_cot_prompt(test_hist, test_map, test_agents, test_meta)
    print("✅ Prompt生成测试通过，Prompt预览:")
    print(prompt[:500] + "...")

    # 测试规则降级
    fallback = RuleBasedFallback()
    action_id, reason = fallback.infer(prompt)
    print(f"\n✅ 规则降级测试通过，动作ID: {action_id}, 原因: {reason}")