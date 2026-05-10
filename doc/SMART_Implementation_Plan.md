# SMART Trial — 12天实现计划
> 目标：5月20日前跑通一个完整encounter（3 stages + re-randomization + trajectory logging）
> 基础：MediQ代码库 + iMEDQA数据集

---

## 确认的基础信息（来自MediQ paper）

| 项目 | 内容 |
|---|---|
| 数据集 | `stellalisy/mediQ`（HuggingFace，CC-BY 4.0） |
| 数据格式 | 初始信息：age + gender + chief complaint；完整record只给Patient System |
| Patient System | Fact-Select变体，factuality 89.1%，直接复用 |
| Expert模型（原版） | GPT-3.5-turbo-0125，部分用GPT-4-turbo |
| 开发阶段替代 | Groq（Llama-3.3-70B，免费）+ Gemini API（免费tier） |
| **Red flags字段** | ⚠️ iMEDQA**没有**，需要我们用LLM离线生成并缓存 |

---

## 项目目录结构

```
smart_trial/
├── config/
│   ├── trial_config.yaml          # 全局配置（模型、阈值、seed等）
│   └── arms/                      # 每个arm的prompt配置
│       ├── stage1_A1a.yaml        # Comprehensive ROS
│       ├── stage1_A1b.yaml        # Focused chief-complaint
│       ├── stage1_A1c.yaml        # Red-flag first
│       ├── stage2_A2a.yaml        # Always retrieve
│       ├── stage2_A2b.yaml        # Confidence-conditional
│       ├── stage2_A2c.yaml        # Parametric only
│       ├── stage3_A3a.yaml        # Single dx + plan
│       ├── stage3_A3b.yaml        # Differential + SDM
│       └── stage3_A3c.yaml        # Escalate
│
├── core/
│   ├── patient_agent.py           # MediQ Fact-Select Patient（直接复用）
│   ├── doctor_agent.py            # MediQ Expert + arm注入层
│   ├── judge.py                   # LLM-judge：计算R1、R2、outcome
│   ├── orchestrator.py            # 主控：stage切换 + re-randomization
│   └── randomizer.py              # 分层随机 + conditional re-randomization
│
├── data/
│   ├── loader.py                  # 加载iMEDQA，标准化格式
│   └── red_flag_cache.json        # 离线生成的red flags（按case_id索引）
│
├── models/
│   └── model_client.py            # 统一模型接口（Groq/Gemini/OpenAI/Anthropic）
│
├── logging/
│   └── trajectory_logger.py       # JSONL格式完整轨迹记录
│
├── scripts/
│   ├── generate_red_flags.py      # 离线为iMEDQA生成red flags
│   └── summarize_encounters.py    # 读JSONL，输出统计摘要
│
├── tests/
│   ├── test_patient.py            # smoke test：Patient能正常回答
│   ├── test_judge.py              # smoke test：R1打分稳定
│   └── test_encounter.py          # 完整encounter端到端测试
│
└── run_encounter.py               # 入口脚本
```

---

## 12天详细计划

### ── Phase 1：地基 Day 1-3（5/8–5/10）──

---

### Day 1 — 环境 + 数据 + 模型接口

**任务清单：**
- [ ] 确认MediQ repo结构，标记哪些文件我们要复用
- [ ] 从HuggingFace下载iMEDQA，确认字段格式
- [ ] 写`models/model_client.py`（统一接口）
- [ ] 申请Groq API key（免费，秒批）

**`models/model_client.py` 骨架：**
```python
import os
from typing import List, Dict, Optional

class ModelClient:
    """
    统一模型调用接口。切换模型只需改provider和model_name。
    支持：groq / gemini / openai / anthropic
    """
    def __init__(self, provider: str, model_name: str, api_key: Optional[str] = None):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
        self._client = self._init_client()

    def _init_client(self):
        if self.provider == "groq":
            from groq import Groq
            return Groq(api_key=self.api_key)
        elif self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            return genai.GenerativeModel(self.model_name)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def chat(self, messages: List[Dict], system_prompt: Optional[str] = None,
             temperature: float = 0.5) -> str:
        """
        统一chat接口。messages格式：[{"role": "user/assistant", "content": "..."}]
        返回：模型回复的字符串
        """
        if self.provider in ["groq", "openai"]:
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=full_messages,
                temperature=temperature
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                system=system_prompt or "",
                messages=messages,
                temperature=temperature
            )
            return response.content[0].text

        elif self.provider == "gemini":
            # Gemini需要不同的格式转换
            chat = self._client.start_chat()
            if system_prompt:
                chat.send_message(system_prompt)
            for msg in messages[:-1]:
                chat.send_message(msg["content"])
            response = chat.send_message(messages[-1]["content"])
            return response.text
```

**`data/loader.py` 骨架：**
```python
from datasets import load_dataset
from typing import List, Dict

def load_imedqa(split: str = "validation", max_cases: int = None) -> List[Dict]:
    """
    加载iMEDQA数据集，转换为我们的标准格式。
    
    原始字段（来自MediQ/MEDQA）：
    - question: 包含完整patient record的MCQ
    - answer: 正确答案选项
    - options: {"A": ..., "B": ..., ...}
    - meta_info: 包含specialty等元信息（如果有）
    
    我们的标准格式：
    - case_id: str
    - chief_complaint: str（从patient record提取）
    - age: str
    - gender: str  
    - full_record: str（完整，只给Patient System）
    - ground_truth_answer: str
    - options: dict
    - case_category: str（从meta_info或关键词推断）
    - red_flags: List[str]（从red_flag_cache加载，初始为空列表）
    """
    dataset = load_dataset("stellalisy/mediQ", split=split)
    if max_cases:
        dataset = dataset.select(range(max_cases))
    
    cases = []
    for i, item in enumerate(dataset):
        case = _parse_medqa_item(item, case_id=f"imedqa_{split}_{i:04d}")
        cases.append(case)
    return cases

def _parse_medqa_item(item: Dict, case_id: str) -> Dict:
    # 从question字段提取patient info
    # MediQ的格式：前几句是patient background，最后是问题
    full_text = item.get("question", "")
    
    return {
        "case_id": case_id,
        "full_record": full_text,
        "chief_complaint": _extract_chief_complaint(full_text),
        "age": _extract_age(full_text),
        "gender": _extract_gender(full_text),
        "ground_truth_answer": item.get("answer", ""),
        "options": item.get("options", {}),
        "case_category": _infer_category(full_text),
        "red_flags": []  # 后续从red_flag_cache.json填充
    }

def _extract_chief_complaint(text: str) -> str:
    # 简单启发式：提取"presents with"之后的内容
    # 20号之前用简单版本，后续可以用LLM提取
    import re
    match = re.search(r'presents? with (.+?)[\.\,]', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # fallback：取第一句话
    return text.split('.')[0].strip()

def _extract_age(text: str) -> str:
    import re
    match = re.search(r'(\d+)[\s-]*year[\s-]*old', text, re.IGNORECASE)
    return match.group(1) if match else "unknown"

def _extract_gender(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["woman", "female", "she", "her"]):
        return "female"
    elif any(w in text_lower for w in ["man", "male", "he", "his", "boy"]):
        return "male"
    return "unknown"

def _infer_category(text: str) -> str:
    # 基于关键词的简单分类，够用于pilot
    text_lower = text.lower()
    if any(w in text_lower for w in ["chest pain", "cardiac", "heart", "MI", "angina"]):
        return "Cardiology"
    elif any(w in text_lower for w in ["abdominal", "nausea", "vomit", "diarrhea", "bowel"]):
        return "GI"
    elif any(w in text_lower for w in ["headache", "seizure", "stroke", "neurolog"]):
        return "Neuro"
    elif any(w in text_lower for w in ["cough", "breath", "lung", "pulmon"]):
        return "Pulm"
    elif any(w in text_lower for w in ["fever", "infect", "bacteria", "virus"]):
        return "Infectious"
    return "Other"
```

**Day 1验收：** `python -c "from data.loader import load_imedqa; cases = load_imedqa(max_cases=5); print(cases[0])"` 能跑出来。

---

### Day 2 — Arm配置文件 + Patient Agent封装

**任务清单：**
- [ ] 写9个arm的YAML配置文件
- [ ] 封装`core/patient_agent.py`（复用MediQ Fact-Select）
- [ ] 写`config/trial_config.yaml`

**`config/trial_config.yaml`：**
```yaml
# 全局配置
trial:
  max_turns: 20
  stage1_turns: 4        # Turn 1-4
  stage2_turns: 6        # Turn 5-10  
  stage3_max_turns: 10   # Turn 11-20
  R1_responder_threshold: 6
  R2_high_confidence_threshold: 0.7

models:
  patient_simulator:
    provider: "groq"
    model_name: "llama-3.3-70b-versatile"
    temperature: 0.3      # 低温度，减少Patient幻觉
  doctor_agent:
    provider: "groq"
    model_name: "llama-3.3-70b-versatile"
    temperature: 0.5
  judge:
    provider: "gemini"
    model_name: "gemini-1.5-flash"  # 免费tier，够用
    temperature: 0.1      # Judge要尽量确定性

randomization:
  seed: 42
  stratify_by: "case_category"

logging:
  output_dir: "outputs/encounters"
  format: "jsonl"
```

**`config/arms/stage1_A1c.yaml` 示例（Red-flag first）：**
```yaml
arm_id: "A1c"
stage: 1
name: "Red-flag first"
max_turns: 4

system_prompt_injection: |
  ## 当前阶段策略（Stage 1 - 信息收集）
  
  在接下来4轮对话中，你必须严格按照以下策略提问：
  
  1. 首先识别病人主诉对应的危险信号（Red flags）清单
     - 胸痛 → 放射到左臂/下颌、冷汗、呼吸困难、晕厥、家族心脏病史
     - 头痛 → 突发剧痛、颈项强直、发热、视觉改变、意识改变
     - 腹痛 → 板状腹、血便、黄疸、体重骤降、持续呕吐
  2. 按照清单逐条排查，每轮问1-2个red flag相关问题
  3. Red flags排查完毕后，再询问其他症状
  4. 每次只问一个具体的、原子性的问题（不要一次问多个）
  
  重要：不要主动透露你的诊断思路，保持自然对话。

tool_access:
  retrieval: false
  calculator: false
```

**`config/arms/stage2_A2b.yaml` 示例（Confidence-conditional）：**
```yaml
arm_id: "A2b"
stage: 2
name: "Confidence-conditional retrieval"
max_turns: 6

system_prompt_injection: |
  ## 当前阶段策略（Stage 2 - 工具使用）
  
  在每一轮对话前，你需要：
  1. 在内心评估你对当前最可能诊断的置信度（0.0-1.0）
  2. 在回复的最开头以固定格式输出：[CONFIDENCE: 0.XX]
  3. 如果置信度 < 0.7，你可以调用检索工具查阅临床指南
  4. 如果置信度 >= 0.7，不调用工具，直接继续问诊
  
  置信度输出格式（必须遵守）：
  [CONFIDENCE: 0.65]
  你的下一个问题...

tool_access:
  retrieval: true
  calculator: false
  retrieval_condition: "confidence < 0.7"
```

**`core/patient_agent.py` 封装：**
```python
from typing import List, Dict
from models.model_client import ModelClient

class PatientAgent:
    """
    封装MediQ的Fact-Select Patient System。
    核心逻辑：把case record分解为atomic facts，
    对每个expert问题只返回相关的facts。
    """
    
    SYSTEM_PROMPT = """你是一个正在就诊的病人。你只能根据你已知的症状和病史来回答医生的问题。
    
规则：
1. 只回答医生问到的问题，不要主动提供额外信息
2. 用普通人的语言回答，不要使用医学术语
3. 如果医生问的问题你的病史里没有相关信息，就说"我不知道"或"没有注意到"
4. 不要推测或编造信息"""

    def __init__(self, model_client: ModelClient, case: Dict):
        self.model = model_client
        self.case = case
        self.atomic_facts = []
        self.conversation_history = []
        self._decompose_facts()
    
    def _decompose_facts(self):
        """将完整病历分解为atomic facts（MediQ Fact-Select核心步骤）"""
        prompt = f"""将以下病人信息分解为独立的原子事实列表。
每个事实只包含一个信息点，但要自给自足。
每行一个事实，用数字编号。

病人信息：
{self.case['full_record']}

请输出原子事实列表："""
        
        response = self.model.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1  # 分解facts要确定性
        )
        
        # 解析编号列表
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                # 去掉编号
                fact = line.split('.', 1)[-1].strip()
                if fact:
                    self.atomic_facts.append(fact)
    
    def respond(self, doctor_question: str) -> str:
        """
        给定医生问题，从atomic facts中选择相关facts并回答。
        这是Fact-Select的核心机制。
        """
        # Step 1: 找到相关的facts
        relevant_facts = self._select_relevant_facts(doctor_question)
        
        # Step 2: 基于facts生成自然语言回答
        if not relevant_facts:
            return "我不太清楚，没有特别注意到这个。"
        
        facts_text = "\n".join(f"- {f}" for f in relevant_facts)
        prompt = f"""基于以下关于这个病人的事实，用普通病人的语言回答医生的问题。
只使用给出的事实，不要推测或添加信息。

病人事实：
{facts_text}

医生问题：{doctor_question}

病人回答（用第一人称，自然语言）："""
        
        response = self.model.chat(
            [{"role": "user", "content": prompt}],
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.3
        )
        
        # 更新对话历史
        self.conversation_history.append({
            "role": "doctor", "content": doctor_question
        })
        self.conversation_history.append({
            "role": "patient", "content": response
        })
        
        return response
    
    def _select_relevant_facts(self, question: str) -> List[str]:
        """从atomic facts中选出回答该问题所需的facts"""
        if not self.atomic_facts:
            return []
        
        facts_numbered = "\n".join(
            f"{i+1}. {f}" for i, f in enumerate(self.atomic_facts)
        )
        
        prompt = f"""从以下病人事实列表中，选出能回答医生问题的事实编号。
最多选2个，如果没有相关事实请回答"无"。
只输出编号，用逗号分隔（例如：1,3）或"无"。

事实列表：
{facts_numbered}

医生问题：{question}

相关事实编号："""
        
        response = self.model.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0
        ).strip()
        
        if response.lower() in ["无", "none", "n/a", ""]:
            return []
        
        selected = []
        for part in response.split(','):
            try:
                idx = int(part.strip()) - 1
                if 0 <= idx < len(self.atomic_facts):
                    selected.append(self.atomic_facts[idx])
            except ValueError:
                pass
        return selected
```

**Day 2验收：** `python tests/test_patient.py` — 给一个case，Patient能对3个不同问题给出合理回答。

---

### Day 3 — Doctor Agent + Arm注入机制

**任务清单：**
- [ ] 写`core/doctor_agent.py`
- [ ] 测试arm切换时conversation history是否正确传递
- [ ] 写smoke test

**`core/doctor_agent.py`：**
```python
import yaml
import re
from typing import Dict, List, Optional, Tuple
from models.model_client import ModelClient

class DoctorAgent:
    """
    封装MediQ Expert System + SMART arm注入层。
    
    核心扩展：
    - 在system prompt里注入当前stage的arm instruction
    - 跨stage保持conversation history
    - 解析confidence（Stage 2用）
    - 识别final conclusion（Stage 3结束条件）
    """
    
    BASE_SYSTEM_PROMPT = """你是一个经验丰富的初级保健（primary care）医生，正在通过文字对话为虚拟病人进行诊断。

你的任务：
1. 通过提问收集病史和症状信息
2. 根据收集到的信息做出诊断判断
3. 最终给出诊断结论和处置建议

基本规则：
- 每次只问一个具体问题
- 保持专业但易于理解的语言
- 不要一次给出太多信息"""

    CONCLUSION_MARKERS = [
        "based on your symptoms",
        "most likely diagnosis",
        "i believe you have",
        "my assessment is",
        "you should go to",
        "emergency department",
        "final diagnosis",
        "[DIAGNOSIS]",
        "[CONCLUSION]"
    ]

    def __init__(self, model_client: ModelClient, initial_arm_config: Dict):
        self.model = model_client
        self.current_arm = initial_arm_config
        self.conversation_history = []  # 完整对话历史，跨stage保持
        self.current_stage = 1
        self.turn_count = 0
        self._last_confidence = None
        self._final_diagnosis = None
        self._has_concluded = False
    
    def switch_arm(self, new_arm_config: Dict):
        """
        Stage切换时更新arm配置。
        重要：conversation_history不清空，保持连续性。
        """
        self.current_arm = new_arm_config
        self.current_stage = new_arm_config["stage"]
    
    def respond(self, patient_message: str) -> Tuple[str, Optional[float]]:
        """
        给定病人回复，生成医生的下一个问题/结论。
        
        返回：(医生回复文本, confidence分数（仅Stage 2有）)
        """
        self.turn_count += 1
        
        # 更新对话历史
        if patient_message:
            self.conversation_history.append({
                "role": "user",
                "content": patient_message
            })
        
        # 构建当前的system prompt（base + arm注入）
        system_prompt = self._build_system_prompt()
        
        # 调用模型
        response = self.model.chat(
            messages=self.conversation_history,
            system_prompt=system_prompt
        )
        
        # 解析confidence（Stage 2）
        confidence = None
        if self.current_stage == 2:
            confidence, response = self._extract_confidence(response)
            self._last_confidence = confidence
        
        # 检查是否给出了最终结论（Stage 3）
        if self.current_stage == 3:
            self._check_conclusion(response)
        
        # 更新对话历史
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response, confidence
    
    def _build_system_prompt(self) -> str:
        """Base prompt + 当前arm的instruction"""
        arm_instruction = self.current_arm.get("system_prompt_injection", "")
        return f"{self.BASE_SYSTEM_PROMPT}\n\n{arm_instruction}"
    
    def _extract_confidence(self, response: str) -> Tuple[Optional[float], str]:
        """从Stage 2的回复中提取[CONFIDENCE: X.XX]标记"""
        pattern = r'\[CONFIDENCE:\s*(0\.\d+|1\.0)\]'
        match = re.search(pattern, response)
        if match:
            confidence = float(match.group(1))
            # 从回复中移除confidence标记
            clean_response = re.sub(pattern, '', response).strip()
            return confidence, clean_response
        return None, response
    
    def _check_conclusion(self, response: str):
        """检查Stage 3的回复是否包含最终结论"""
        response_lower = response.lower()
        for marker in self.CONCLUSION_MARKERS:
            if marker.lower() in response_lower:
                self._has_concluded = True
                self._final_diagnosis = response
                break
    
    def get_initial_message(self, case: Dict) -> str:
        """
        生成第一轮的医生开场白，呈现chief complaint。
        这是encounter的第一句话。
        """
        age = case.get("age", "unknown")
        gender = case.get("gender", "unknown")
        chief_complaint = case.get("chief_complaint", "some symptoms")
        
        initial_msg = (f"Hello, I'm your doctor today. I see you're a "
                      f"{age}-year-old {gender} presenting with {chief_complaint}. "
                      f"I'd like to ask you a few questions to better understand "
                      f"your condition.")
        
        self.conversation_history.append({
            "role": "assistant",
            "content": initial_msg
        })
        return initial_msg
    
    def has_concluded(self) -> bool:
        return self._has_concluded
    
    def get_final_diagnosis(self) -> Optional[str]:
        return self._final_diagnosis
    
    def get_last_confidence(self) -> Optional[float]:
        return self._last_confidence


def load_arm_config(arm_id: str, arms_dir: str = "config/arms") -> Dict:
    """从YAML文件加载arm配置"""
    filepath = f"{arms_dir}/{arm_id.replace('A', 'stage')}.yaml"
    # 映射arm_id到文件名
    stage_map = {
        "A1a": "stage1_A1a", "A1b": "stage1_A1b", "A1c": "stage1_A1c",
        "A2a": "stage2_A2a", "A2b": "stage2_A2b", "A2c": "stage2_A2c",
        "A3a": "stage3_A3a", "A3b": "stage3_A3b", "A3c": "stage3_A3c",
    }
    filename = stage_map.get(arm_id, arm_id)
    filepath = f"{arms_dir}/{filename}.yaml"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
```

**Day 3验收：** Doctor能完成一轮对话，arm切换后conversation history不丢失。

---

### ── Phase 2：核心逻辑 Day 4-7（5/11–5/14）──

---

### Day 4 — Judge模块（R1计算）

**这是最关键的部分，要写得仔细。**

**`core/judge.py`：**
```python
import re
import json
from typing import Dict, List, Tuple
from models.model_client import ModelClient

class StageJudge:
    """
    LLM-judge，负责：
    1. 计算R1（Stage 1结束时的信息充分性分数）
    2. 计算R2（Stage 2结束时的置信度状态）
    3. 评估最终outcome（诊断准确性、red flag遗漏、危险建议）
    """
    
    R1_RUBRIC_PROMPT = """你是一个医学教育评估专家。请评估以下医患对话中，医生收集病史信息的充分性。

病人主诉：{chief_complaint}
病人年龄：{age}，性别：{gender}

对话记录：
{conversation}

请从以下5个维度评分，每个维度0-2分：

1. **OPQRST完整性**（0-2分）
   - 0分：几乎没有问OPQRST
   - 1分：问了部分（onset或severity等）
   - 2分：问了多个维度（发作时间、性质、严重程度、诱因等）

2. **Red flags排查**（0-2分）
   - 0分：没有问任何危险信号
   - 1分：问了1-2个相关red flag
   - 2分：系统性问了与主诉相关的主要red flags

3. **相关既往史**（0-2分）
   - 0分：没问过去病史
   - 1分：问了既往史但不完整
   - 2分：问了与主诉相关的关键既往病史

4. **用药史/过敏史**（0-2分）
   - 0分：完全没问
   - 1分：问了其中一个
   - 2分：两个都问了

5. **社会史/家族史**（0-2分）
   - 0分：没问
   - 1分：问了其中一个
   - 2分：两个都问了（吸烟、饮酒、家族病史等）

请严格按照以下JSON格式输出，不要输出其他内容：
{{
  "OPQRST": <0-2的整数>,
  "red_flags": <0-2的整数>,
  "past_medical_history": <0-2的整数>,
  "medications_allergies": <0-2的整数>,
  "social_family_history": <0-2的整数>,
  "total": <总分0-10>,
  "responder": <true如果total>=6，否则false>,
  "reasoning": "<一句话说明打分依据>"
}}"""

    OUTCOME_PROMPT = """请评估以下医疗对话的质量。

病人主诉：{chief_complaint}
正确诊断（ground truth）：{ground_truth}
医生最终给出的诊断/结论：{final_diagnosis}

请判断：
1. 诊断是否正确（correct: true/false）
   - true：医生给出的诊断与正确诊断吻合，或包含正确诊断作为主要选项
   - false：诊断明显错误或偏离
   
2. 是否有危险建议（dangerous_advice: true/false）
   - true：医生建议了违反医学常识的做法（如让NSTEMI病人回家观察）
   - false：建议合理或偏保守

3. 是否建议了正确的处置（appropriate_management: true/false）

请严格按以下JSON格式输出：
{{
  "diag_correct": <true/false>,
  "dangerous_advice": <true/false>,
  "appropriate_management": <true/false>,
  "reasoning": "<简要说明>"
}}"""

    def __init__(self, model_client: ModelClient):
        self.model = model_client
    
    def compute_R1(self, conversation_history: List[Dict],
                   case: Dict) -> Dict:
        """
        计算Stage 1结束时的信息充分性分数。
        输入：前4轮对话 + case基本信息
        输出：R1分数字典，含total和responder标志
        """
        # 格式化对话记录
        conv_text = self._format_conversation(conversation_history)
        
        prompt = self.R1_RUBRIC_PROMPT.format(
            chief_complaint=case.get("chief_complaint", "unknown"),
            age=case.get("age", "unknown"),
            gender=case.get("gender", "unknown"),
            conversation=conv_text
        )
        
        response = self.model.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1  # 尽量确定性
        )
        
        return self._parse_json_response(response, default={
            "OPQRST": 0, "red_flags": 0, "past_medical_history": 0,
            "medications_allergies": 0, "social_family_history": 0,
            "total": 0, "responder": False, "reasoning": "parse error"
        })
    
    def compute_R2(self, conversation_history: List[Dict],
                   confidence_scores: List[float]) -> Dict:
        """
        计算Stage 2结束时的置信度状态。
        
        注意：R2在Stage 2结束时只能判断high/low confidence。
        correct/wrong要等encounter结束后用compute_outcome补充。
        
        输入：Stage 2期间的对话 + doctor报告的confidence列表
        输出：R2状态字典
        """
        # 用Stage 2最后一次confidence作为R2
        if confidence_scores:
            final_confidence = confidence_scores[-1]
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            # 如果arm是A2a或A2c（没有confidence reporting），给默认值
            final_confidence = 0.5
            avg_confidence = 0.5
        
        # 配置里的threshold是0.7
        is_high_confidence = final_confidence >= 0.7
        
        return {
            "final_confidence": final_confidence,
            "avg_confidence": avg_confidence,
            "confidence_level": "high" if is_high_confidence else "low",
            "confidence_scores": confidence_scores,
            # correct字段在encounter结束后由compute_outcome填充
            "R2_category": None  # "high-correct"/"high-wrong"/"low-confidence"
        }
    
    def evaluate_outcome(self, final_diagnosis: str,
                        case: Dict,
                        conversation_history: List[Dict],
                        R2: Dict) -> Dict:
        """
        Encounter结束后评估最终outcome，同时补充R2的correct信息。
        """
        if not final_diagnosis:
            return {
                "diag_correct": False,
                "dangerous_advice": False,
                "appropriate_management": False,
                "red_flag_miss": self._check_red_flag_miss(
                    conversation_history, case.get("red_flags", [])
                ),
                "reasoning": "no final diagnosis provided"
            }
        
        prompt = self.OUTCOME_PROMPT.format(
            chief_complaint=case.get("chief_complaint", "unknown"),
            ground_truth=case.get("ground_truth_answer", "unknown"),
            final_diagnosis=final_diagnosis
        )
        
        response = self.model.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        outcome = self._parse_json_response(response, default={
            "diag_correct": False,
            "dangerous_advice": False,
            "appropriate_management": False,
            "reasoning": "parse error"
        })
        
        # 检查red flag遗漏
        outcome["red_flag_miss"] = self._check_red_flag_miss(
            conversation_history, case.get("red_flags", [])
        )
        
        # 补充R2的correct信息
        if R2["confidence_level"] == "high":
            R2["R2_category"] = (
                "high-correct" if outcome["diag_correct"] else "high-wrong"
            )
        else:
            R2["R2_category"] = "low-confidence"
        
        return outcome
    
    def _check_red_flag_miss(self, conversation: List[Dict],
                              red_flags: List[str]) -> bool:
        """
        检查是否有red flag被遗漏。
        简单实现：看conversation里是否提到了各个red flag关键词。
        """
        if not red_flags:
            return False  # 没有red flags定义，不算miss
        
        conv_text = " ".join(
            msg.get("content", "").lower()
            for msg in conversation
        )
        
        missed = []
        for flag in red_flags:
            # 简单关键词匹配（pilot够用）
            flag_keywords = flag.lower().split()
            if not any(kw in conv_text for kw in flag_keywords):
                missed.append(flag)
        
        return len(missed) > 0  # True = 有遗漏
    
    def _format_conversation(self, history: List[Dict]) -> str:
        """将conversation history格式化为可读文本"""
        lines = []
        for msg in history:
            role = "Doctor" if msg["role"] == "assistant" else "Patient"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
    
    def _parse_json_response(self, response: str, default: Dict) -> Dict:
        """解析LLM返回的JSON，带fallback"""
        try:
            # 有时LLM会在JSON前后加```
            clean = re.sub(r'```json\s*|\s*```', '', response).strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            # 尝试提取JSON块
            match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            return default
```

**Day 4验收：** 给一段假的对话，R1能返回合理的JSON分数。

---

### Day 5 — Randomizer

**`core/randomizer.py`：**
```python
import random
from typing import Dict, List

class TrialRandomizer:
    """
    SMART re-randomization逻辑。
    
    核心：根据R_t（中间结果）决定下一stage的可用arm池。
    所有随机seed提前固定，保证可重现。
    """
    
    # Stage 2的arm池，conditional on R1
    STAGE2_POOLS = {
        "responder":     ["A2a", "A2b", "A2c"],  # R1 >= 6
        "non-responder": ["A2a", "A2b"],           # R1 < 6，排除A2c
    }
    
    # Stage 3的arm池，conditional on R2 confidence level
    # 注意：此处只用confidence_level（high/low），不用correct/wrong
    # correct/wrong是事后信息，不能用于实时randomization
    STAGE3_POOLS = {
        "high": ["A3a", "A3b"],     # 高置信度 → 可以给单一诊断
        "low":  ["A3b", "A3c"],     # 低置信度 → 不给单一诊断
    }
    
    # Stage 1永远是全池
    STAGE1_POOL = ["A1a", "A1b", "A1c"]
    
    def __init__(self, seed: int, stratify_by: str = "case_category"):
        self.seed = seed
        self.stratify_by = stratify_by
        # 用case特定的seed，保证每个case的randomization独立可重现
        self._rng = random.Random(seed)
    
    def assign_stage1_arm(self, case: Dict) -> str:
        """
        Stage 1 randomization：按case_category分层，均匀随机。
        """
        category = case.get(self.stratify_by, "Other")
        # 用case_id + category生成确定性的随机选择
        case_seed = hash(f"{self.seed}_{case['case_id']}_stage1")
        rng = random.Random(case_seed)
        arm = rng.choice(self.STAGE1_POOL)
        return arm
    
    def assign_stage2_arm(self, case: Dict, R1: Dict) -> Dict:
        """
        Stage 2 re-randomization：conditional on R1 responder status。
        
        返回：{"arm": arm_id, "pool_used": pool_name, "R1_total": score}
        """
        is_responder = R1.get("responder", False)
        pool_key = "responder" if is_responder else "non-responder"
        pool = self.STAGE2_POOLS[pool_key]
        
        case_seed = hash(f"{self.seed}_{case['case_id']}_stage2")
        rng = random.Random(case_seed)
        arm = rng.choice(pool)
        
        return {
            "arm": arm,
            "pool_used": pool_key,
            "pool": pool,
            "R1_total": R1.get("total", 0)
        }
    
    def assign_stage3_arm(self, case: Dict, R2: Dict) -> Dict:
        """
        Stage 3 re-randomization：conditional on R2 confidence level。
        
        注意：这里只用confidence_level（high/low），
        不是R2_category（后者包含correct/wrong，是事后信息）。
        """
        confidence_level = R2.get("confidence_level", "low")
        pool = self.STAGE3_POOLS.get(confidence_level, ["A3b", "A3c"])
        
        case_seed = hash(f"{self.seed}_{case['case_id']}_stage3")
        rng = random.Random(case_seed)
        arm = rng.choice(pool)
        
        return {
            "arm": arm,
            "pool_used": confidence_level,
            "pool": pool,
            "R2_confidence": R2.get("final_confidence", None)
        }
    
    def get_assignment_summary(self, stage1_result, stage2_result,
                               stage3_result) -> Dict:
        """生成本次encounter的完整randomization摘要"""
        return {
            "stage1_arm": stage1_result,
            "stage2_arm": stage2_result["arm"],
            "stage2_pool": stage2_result["pool_used"],
            "stage3_arm": stage3_result["arm"],
            "stage3_pool": stage3_result["pool_used"],
            "trajectory_id": f"{stage1_result}→{stage2_result['arm']}→{stage3_result['arm']}"
        }
```

**Day 5验收：** 用几个mock R1/R2值，验证arm分配逻辑正确（responder进3个池，non-responder进2个池）。

---

### Day 6 — Trajectory Logger

**`logging/trajectory_logger.py`：**
```python
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

class TrajectoryLogger:
    """
    记录完整的encounter轨迹到JSONL格式。
    每个encounter一条记录，包含：
    - 基本信息（case_id, arms, R scores）
    - 完整对话历史（每轮）
    - Outcome结果
    """
    
    def __init__(self, output_dir: str = "outputs/encounters"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._current = {}
        self._turns = []
        self._stage2_confidences = []
    
    def start_encounter(self, case: Dict, seed: int, stage1_arm: str):
        """初始化一个新的encounter记录"""
        self._turns = []
        self._stage2_confidences = []
        self._current = {
            "encounter_id": f"enc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "case_id": case["case_id"],
            "case_category": case.get("case_category", "Other"),
            "chief_complaint": case.get("chief_complaint", ""),
            "ground_truth": case.get("ground_truth_answer", ""),
            "seed": seed,
            "stage1_arm": stage1_arm,
            # 以下字段在后续log_stage_transition时填充
            "R1": None,
            "stage2_arm": None,
            "R2": None,
            "stage3_arm": None,
            "outcome": None,
            "total_turns": 0,
            "timestamp_start": datetime.now().isoformat(),
            "trajectory": []
        }
    
    def log_turn(self, turn_number: int, stage: int, arm_id: str,
                 doctor_message: str, patient_message: str,
                 confidence: Optional[float] = None):
        """记录单轮对话"""
        turn_record = {
            "turn": turn_number,
            "stage": stage,
            "arm": arm_id,
            "doctor": doctor_message,
            "patient": patient_message,
        }
        if confidence is not None:
            turn_record["confidence"] = confidence
            self._stage2_confidences.append(confidence)
        
        self._turns.append(turn_record)
        self._current["total_turns"] = turn_number
    
    def log_stage_transition(self, stage: int, R_score: Dict,
                              next_arm: str, pool_info: Dict):
        """记录stage切换点"""
        if stage == 1:
            self._current["R1"] = R_score
            self._current["stage2_arm"] = next_arm
            self._current["stage2_pool"] = pool_info.get("pool_used")
        elif stage == 2:
            self._current["R2"] = R_score
            self._current["stage3_arm"] = next_arm
            self._current["stage3_pool"] = pool_info.get("pool_used")
    
    def get_stage2_confidences(self) -> List[float]:
        return self._stage2_confidences
    
    def finalize(self, outcome: Dict) -> Dict:
        """完成encounter记录，写入JSONL文件"""
        self._current["outcome"] = outcome
        self._current["trajectory"] = self._turns
        self._current["timestamp_end"] = datetime.now().isoformat()
        
        # 写入JSONL
        output_file = os.path.join(
            self.output_dir,
            f"{self._current['case_id']}.jsonl"
        )
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(self._current, ensure_ascii=False) + '\n')
        
        return self._current
    
    @staticmethod
    def load_all_encounters(output_dir: str) -> List[Dict]:
        """加载所有encounter记录（供分析用）"""
        encounters = []
        for fname in os.listdir(output_dir):
            if fname.endswith('.jsonl'):
                with open(os.path.join(output_dir, fname), 'r') as f:
                    for line in f:
                        if line.strip():
                            encounters.append(json.loads(line))
        return encounters
```

**Day 6验收：** 写一个mock encounter，检查JSONL格式是否完整。

---

### Day 7 — Orchestrator（把所有组件串起来）

**`core/orchestrator.py`：**
```python
import yaml
from typing import Dict, Optional
from models.model_client import ModelClient
from core.patient_agent import PatientAgent
from core.doctor_agent import DoctorAgent, load_arm_config
from core.judge import StageJudge
from core.randomizer import TrialRandomizer
from logging.trajectory_logger import TrajectoryLogger

class TrialOrchestrator:
    """
    SMART Trial主控。
    负责：加载case → stage管理 → re-randomization → 记录轨迹
    
    Stage结构（当前版本：固定轮次）：
    - Stage 1: Turn 1-4   （信息收集）
    - Stage 2: Turn 5-10  （工具使用）
    - Stage 3: Turn 11-20 （给结论，直到has_concluded或到第20轮）
    """
    
    def __init__(self, config_path: str = "config/trial_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化各组件的模型client
        self.patient_model = self._make_client("patient_simulator")
        self.doctor_model = self._make_client("doctor_agent")
        self.judge_model = self._make_client("judge")
        
        self.judge = StageJudge(self.judge_model)
    
    def run_encounter(self, case: Dict, seed: Optional[int] = None) -> Dict:
        """
        运行一个完整的encounter。
        
        输入：标准化的case字典
        输出：完整的trajectory记录（同时写入JSONL）
        """
        if seed is None:
            seed = self.config["randomization"]["seed"]
        
        randomizer = TrialRandomizer(seed)
        logger = TrajectoryLogger(self.config["logging"]["output_dir"])
        
        # ── Stage 1 Randomization ──
        stage1_arm_id = randomizer.assign_stage1_arm(case)
        stage1_arm = load_arm_config(stage1_arm_id)
        
        print(f"\n{'='*60}")
        print(f"Case: {case['case_id']} | {case['case_category']}")
        print(f"Chief Complaint: {case['chief_complaint']}")
        print(f"Stage 1 Arm: {stage1_arm_id} ({stage1_arm['name']})")
        print(f"{'='*60}\n")
        
        # 初始化agents
        doctor = DoctorAgent(self.doctor_model, stage1_arm)
        patient = PatientAgent(self.patient_model, case)
        
        # 开始记录
        logger.start_encounter(case, seed, stage1_arm_id)
        
        # 医生开场白
        initial_msg = doctor.get_initial_message(case)
        print(f"[Turn 0 - Opening]\nDoctor: {initial_msg}\n")
        
        # ── Stage 1: Turn 1-4 ──
        print(f"--- STAGE 1 ({stage1_arm['name']}) ---")
        last_patient_response = initial_msg  # 第一轮patient响应医生开场
        
        for turn in range(1, 5):
            # Doctor问问题
            doctor_msg, confidence = doctor.respond(last_patient_response)
            # Patient回答
            patient_response = patient.respond(doctor_msg)
            
            print(f"[Turn {turn}]")
            print(f"Doctor: {doctor_msg}")
            print(f"Patient: {patient_response}\n")
            
            logger.log_turn(turn, 1, stage1_arm_id,
                          doctor_msg, patient_response, confidence)
            last_patient_response = patient_response
        
        # ── 计算R1，Re-randomize Stage 2 ──
        print("--- Computing R1 ---")
        R1 = self.judge.compute_R1(doctor.conversation_history, case)
        print(f"R1 Score: {R1['total']}/10 | "
              f"Responder: {R1['responder']}")
        
        stage2_assignment = randomizer.assign_stage2_arm(case, R1)
        stage2_arm_id = stage2_assignment["arm"]
        stage2_arm = load_arm_config(stage2_arm_id)
        
        print(f"Stage 2 Arm: {stage2_arm_id} ({stage2_arm['name']}) "
              f"[Pool: {stage2_assignment['pool_used']}]\n")
        
        logger.log_stage_transition(1, R1, stage2_arm_id, stage2_assignment)
        
        # ── Stage 2: Turn 5-10 ──
        doctor.switch_arm(stage2_arm)
        print(f"--- STAGE 2 ({stage2_arm['name']}) ---")
        
        for turn in range(5, 11):
            doctor_msg, confidence = doctor.respond(last_patient_response)
            patient_response = patient.respond(doctor_msg)
            
            conf_str = f" [conf={confidence:.2f}]" if confidence else ""
            print(f"[Turn {turn}]{conf_str}")
            print(f"Doctor: {doctor_msg}")
            print(f"Patient: {patient_response}\n")
            
            logger.log_turn(turn, 2, stage2_arm_id,
                          doctor_msg, patient_response, confidence)
            last_patient_response = patient_response
        
        # ── 计算R2，Re-randomize Stage 3 ──
        print("--- Computing R2 ---")
        stage2_confidences = logger.get_stage2_confidences()
        R2 = self.judge.compute_R2(doctor.conversation_history,
                                    stage2_confidences)
        print(f"R2 Confidence: {R2['confidence_level']} "
              f"(final={R2['final_confidence']:.2f})")
        
        stage3_assignment = randomizer.assign_stage3_arm(case, R2)
        stage3_arm_id = stage3_assignment["arm"]
        stage3_arm = load_arm_config(stage3_arm_id)
        
        print(f"Stage 3 Arm: {stage3_arm_id} ({stage3_arm['name']}) "
              f"[Pool: {stage3_assignment['pool_used']}]\n")
        
        logger.log_stage_transition(2, R2, stage3_arm_id, stage3_assignment)
        
        # ── Stage 3: Turn 11-20 ──
        doctor.switch_arm(stage3_arm)
        print(f"--- STAGE 3 ({stage3_arm['name']}) ---")
        
        turn = 11
        while not doctor.has_concluded() and turn <= 20:
            doctor_msg, confidence = doctor.respond(last_patient_response)
            patient_response = patient.respond(doctor_msg)
            
            print(f"[Turn {turn}]")
            print(f"Doctor: {doctor_msg}")
            if not doctor.has_concluded():
                print(f"Patient: {patient_response}")
            print()
            
            logger.log_turn(turn, 3, stage3_arm_id,
                          doctor_msg, patient_response, confidence)
            last_patient_response = patient_response
            turn += 1
            
            if doctor.has_concluded():
                break
        
        # ── Outcome Measurement ──
        print("--- Evaluating Outcome ---")
        final_diag = doctor.get_final_diagnosis()
        outcome = self.judge.evaluate_outcome(
            final_diagnosis=final_diag,
            case=case,
            conversation_history=doctor.conversation_history,
            R2=R2
        )
        
        print(f"Diagnosis Correct: {outcome['diag_correct']}")
        print(f"Red Flag Miss: {outcome['red_flag_miss']}")
        print(f"Dangerous Advice: {outcome['dangerous_advice']}")
        print(f"Turns Used: {logger._current['total_turns']}")
        
        # ── 写入轨迹 ──
        trajectory = logger.finalize(outcome)
        
        print(f"\n{'='*60}")
        print(f"Trajectory: {trajectory['stage1_arm']} → "
              f"{trajectory['stage2_arm']} → {trajectory['stage3_arm']}")
        print(f"Saved to: outputs/encounters/{case['case_id']}.jsonl")
        print(f"{'='*60}\n")
        
        return trajectory
    
    def _make_client(self, role: str) -> ModelClient:
        cfg = self.config["models"][role]
        return ModelClient(
            provider=cfg["provider"],
            model_name=cfg["model_name"],
            temperature=cfg.get("temperature", 0.5)
        )
```

**Day 7验收：** `python run_encounter.py --case_id imedqa_validation_0042 --seed 42` 跑完一个encounter，输出完整的stage切换和最终JSONL。

---

### ── Phase 3：集成测试 Day 8-10（5/15–5/17）──

---

### Day 8 — Red Flag生成脚本 + 入口脚本

**关于Red Flags的处理策略：**

iMEDQA没有red flags字段。两个选项：

**选项A（推荐用于20号demo）：** 离线用LLM为每个case生成red flags，缓存到JSON。
```python
# scripts/generate_red_flags.py
"""
为iMEDQA的每个case离线生成red flag列表，缓存到data/red_flag_cache.json。
只需要跑一次。
"""
import json
from data.loader import load_imedqa
from models.model_client import ModelClient

PROMPT = """给定病人主诉，列出这个主诉最重要的3-5个危险信号（red flags）。
这些是医生必须问到的、可能提示严重疾病的症状或体征。

主诉：{chief_complaint}
病人年龄：{age}，性别：{gender}

请输出JSON格式：
{{"red_flags": ["red flag 1", "red flag 2", ...]}}"""

def generate_red_flags(cases, model_client, output_path):
    cache = {}
    for case in cases:
        prompt = PROMPT.format(
            chief_complaint=case["chief_complaint"],
            age=case["age"],
            gender=case["gender"]
        )
        response = model_client.chat([{"role": "user", "content": prompt}])
        try:
            data = json.loads(response)
            cache[case["case_id"]] = data.get("red_flags", [])
        except:
            cache[case["case_id"]] = []
        print(f"Generated red flags for {case['case_id']}")
    
    with open(output_path, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"Saved to {output_path}")
```

**选项B（更快但粗糙）：** 在loader里根据case_category给通用red flags。先用这个跑通pipeline，后续再精细化。

```python
# 在data/loader.py里加：
CATEGORY_RED_FLAGS = {
    "Cardiology": ["radiation to left arm or jaw", "diaphoresis",
                   "shortness of breath", "syncope", "family history of heart disease"],
    "Neuro":      ["sudden onset severe headache", "neck stiffness",
                   "focal neurological deficit", "altered consciousness"],
    "GI":         ["blood in stool", "severe abdominal rigidity",
                   "jaundice", "rapid weight loss"],
    "Pulm":       ["hemoptysis", "severe dyspnea at rest",
                   "cyanosis", "oxygen saturation < 90%"],
    "Infectious": ["high fever > 39C", "petechiae",
                   "altered mental status", "rigors"],
    "Other":      ["severe pain", "rapid deterioration"]
}
```

**`run_encounter.py`（入口脚本）：**
```python
#!/usr/bin/env python3
"""
SMART Trial — Single Encounter Runner

用法：
  python run_encounter.py                           # 随机选一个case
  python run_encounter.py --case_id imedqa_val_0042 # 指定case
  python run_encounter.py --seed 123                # 指定seed
  python run_encounter.py --n 5                     # 跑5个encounter
"""
import argparse
import random
from data.loader import load_imedqa, load_red_flag_cache
from core.orchestrator import TrialOrchestrator

def main():
    parser = argparse.ArgumentParser(description="Run SMART Trial Encounter")
    parser.add_argument("--case_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=1, help="Number of encounters")
    parser.add_argument("--config", type=str, default="config/trial_config.yaml")
    args = parser.parse_args()
    
    # 加载数据
    print("Loading iMEDQA dataset...")
    cases = load_imedqa(split="validation", max_cases=100)
    
    # 加载red flags
    try:
        red_flag_cache = load_red_flag_cache("data/red_flag_cache.json")
        for case in cases:
            case["red_flags"] = red_flag_cache.get(case["case_id"], [])
    except FileNotFoundError:
        print("⚠️  red_flag_cache.json not found, using category defaults")
    
    # 初始化orchestrator
    orchestrator = TrialOrchestrator(args.config)
    
    # 选择case
    if args.case_id:
        selected = [c for c in cases if c["case_id"] == args.case_id]
        if not selected:
            print(f"Case {args.case_id} not found")
            return
        run_cases = selected
    else:
        run_cases = random.sample(cases, min(args.n, len(cases)))
    
    # 跑encounter
    results = []
    for case in run_cases:
        print(f"\nRunning encounter for case: {case['case_id']}")
        trajectory = orchestrator.run_encounter(case, seed=args.seed)
        results.append(trajectory)
    
    # 简单摘要
    if len(results) > 1:
        n_correct = sum(1 for r in results
                       if r.get("outcome", {}).get("diag_correct", False))
        print(f"\n{'='*60}")
        print(f"Summary: {n_correct}/{len(results)} correct diagnoses")
        trajectories = [r.get("stage1_arm","?") + "→" + 
                       r.get("stage2_arm","?") + "→" + 
                       r.get("stage3_arm","?")
                       for r in results]
        from collections import Counter
        print("Trajectory distribution:", dict(Counter(trajectories)))

if __name__ == "__main__":
    main()
```

**Day 8验收：** `python run_encounter.py --n 1` 能跑，JSONL文件被创建。

---

### Day 9 — 端到端测试，修Bug

**重点检查清单：**

```
□ Arm切换时，conversation history正确传递（不清空，不重复）
□ R1 judge返回valid JSON（不报ParseError）
□ Stage 2的confidence被正确提取（[CONFIDENCE: 0.XX]格式）
□ Stage 3的has_concluded()在合适时机被触发
□ JSONL文件字段完整（R1、R2、三个arm都有）
□ 最多20轮的上限被正确执行
□ 免费API的rate limit不会导致程序崩溃（加retry逻辑）
```

**最重要的Bug预防 — 加retry和rate limit处理：**
```python
# 在model_client.py里加
import time

def chat_with_retry(self, messages, system_prompt=None,
                    temperature=0.5, max_retries=3):
    for attempt in range(max_retries):
        try:
            return self.chat(messages, system_prompt, temperature)
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                print(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

**Day 9验收：** 完整跑3个不同category的case，没有崩溃，3个JSONL记录完整。

---

### Day 10 — 摘要脚本 + 输出优化

**`scripts/summarize_encounters.py`：**
```python
"""
读取所有JSONL记录，输出统计摘要。
给教授开会用的简洁报告。
"""
import json
import os
from collections import Counter, defaultdict

def summarize(output_dir: str = "outputs/encounters"):
    encounters = []
    for fname in os.listdir(output_dir):
        if fname.endswith('.jsonl'):
            with open(os.path.join(output_dir, fname)) as f:
                for line in f:
                    if line.strip():
                        encounters.append(json.loads(line))
    
    if not encounters:
        print("No encounters found.")
        return
    
    print(f"\n{'='*60}")
    print(f"SMART Trial Summary — {len(encounters)} encounters")
    print(f"{'='*60}")
    
    # Arm分布
    s1_arms = Counter(e["stage1_arm"] for e in encounters)
    s2_arms = Counter(e.get("stage2_arm", "?") for e in encounters)
    s3_arms = Counter(e.get("stage3_arm", "?") for e in encounters)
    print(f"\nStage 1 arm distribution: {dict(s1_arms)}")
    print(f"Stage 2 arm distribution: {dict(s2_arms)}")
    print(f"Stage 3 arm distribution: {dict(s3_arms)}")
    
    # R1分布
    R1_scores = [e["R1"]["total"] for e in encounters if e.get("R1")]
    if R1_scores:
        responder_rate = sum(1 for s in R1_scores if s >= 6) / len(R1_scores)
        print(f"\nR1 scores: mean={sum(R1_scores)/len(R1_scores):.1f}, "
              f"responder_rate={responder_rate:.1%}")
    
    # Outcome
    outcomes = [e.get("outcome", {}) for e in encounters if e.get("outcome")]
    if outcomes:
        n_correct = sum(1 for o in outcomes if o.get("diag_correct", False))
        n_dangerous = sum(1 for o in outcomes if o.get("dangerous_advice", False))
        n_rf_miss = sum(1 for o in outcomes if o.get("red_flag_miss", False))
        print(f"\nOutcomes:")
        print(f"  Diagnostic accuracy: {n_correct}/{len(outcomes)} = "
              f"{n_correct/len(outcomes):.1%}")
        print(f"  Dangerous advice: {n_dangerous}/{len(outcomes)}")
        print(f"  Red flag miss: {n_rf_miss}/{len(outcomes)}")
    
    # 轨迹分布
    trajectories = Counter(
        f"{e.get('stage1_arm','?')}→{e.get('stage2_arm','?')}→{e.get('stage3_arm','?')}"
        for e in encounters
    )
    print(f"\nTop trajectories:")
    for traj, count in trajectories.most_common(5):
        print(f"  {traj}: {count}x")
    
    print(f"\nAvg turns used: "
          f"{sum(e.get('total_turns',0) for e in encounters)/len(encounters):.1f}")

if __name__ == "__main__":
    summarize()
```

---

### ── Phase 4：收尾 Day 11-12（5/18–5/20）──

---

### Day 11 — Arm Prompt精细化 + README

这一天把所有9个arm的YAML prompt认真写完整。
Stage 1和Stage 2的prompt前面已经给了示例，需要补全的是：

**`stage3_A3a.yaml`（Single dx + plan）：**
```yaml
arm_id: "A3a"
stage: 3
name: "Single diagnosis + management plan"

system_prompt_injection: |
  ## 当前阶段策略（Stage 3 - 给出结论）
  
  基于你收集到的所有信息，现在给出：
  1. 一个最可能的诊断（不是鉴别诊断列表，是单一最优诊断）
  2. 具体的处置建议（开药/转诊/急诊/观察，选其中最合适的）
  3. 对病人的清晰解释（用非医学语言）
  
  输出格式：
  [DIAGNOSIS] 你的诊断
  
  基于你描述的症状，我认为...（解释）
  
  关于下一步，我建议...（处置）
  
  如果出现以下情况请立即就医：...（红旗症状提示）
```

**`stage3_A3b.yaml`（Differential + SDM）：**
```yaml
arm_id: "A3b"
stage: 3
name: "Differential diagnosis + shared decision making"

system_prompt_injection: |
  ## 当前阶段策略（Stage 3 - 鉴别诊断+共同决策）
  
  基于你收集到的信息，给出：
  1. Top 3 鉴别诊断，每个附上支持和反对的证据
  2. 邀请病人参与决策：解释不同选项的利弊
  3. 根据病人偏好给出最终建议
  
  输出格式：
  [DIAGNOSIS] 鉴别诊断（最可能→最不可能）
  
  根据你的症状，可能的原因有几个：
  1. [最可能] 因为...
  2. [其次] 因为...
  3. [也可能] 因为...
  
  为了进一步确认，我们可以选择：
  - 选项A：... （优点/缺点）
  - 选项B：... （优点/缺点）
  
  你倾向于哪种方式？
```

**`stage3_A3c.yaml`（Escalate）：**
```yaml
arm_id: "A3c"
stage: 3
name: "Escalate to in-person / emergency"

system_prompt_injection: |
  ## 当前阶段策略（Stage 3 - 建议转诊/急诊）
  
  基于你收集到的信息，如果存在任何不确定性或潜在严重情况：
  1. 不给最终诊断
  2. 清楚告知病人需要立即面诊或急诊
  3. 解释原因（不引起不必要恐慌，但要清晰）
  4. 告知等待期间注意事项
  
  输出格式：
  [DIAGNOSIS] 需要进一步评估
  
  根据你描述的症状，我认为需要...
  
  请[立即去急诊/尽快预约面诊]，因为...
  
  在等待就医期间，请注意...
```

### Day 12 — Buffer + Demo准备

演示脚本（给教授开会用）：
```bash
# 演示命令
python run_encounter.py --case_id imedqa_validation_0042 --seed 42
python scripts/summarize_encounters.py
```

准备一张流程图，展示：
```
Case加载 → Stage 1 Randomization(A1a/b/c)
        → Turn 1-4对话
        → R1计算（LLM-judge）
        → Stage 2 Re-randomization（conditional on R1 responder状态）
        → Turn 5-10对话
        → R2计算
        → Stage 3 Re-randomization（conditional on R2 confidence）
        → Turn 11-20对话（直到conclusion）
        → Outcome评估
        → JSONL记录
```

---

## 关键设计决策备忘

| 决策 | 当前选择 | 原因 | 待确认 |
|---|---|---|---|
| Stage切换 | 固定轮次（1-4, 5-10, 11-20） | 简单，可重现，符合教授方案 | 后续可改为adaptive |
| Red flags | 先用category级别通用版，后补精细版 | 20号前来得及 | 需要confirm |
| R2时序问题 | Stage 2结束时只用confidence(high/low)，correct/wrong事后补充 | 解决了时序矛盾 | 已确认 |
| 免费模型 | Groq(Llama-3.3-70B) + Gemini Flash | 免费够pilot | 实测后换Claude/GPT |
| Patient System | 直接复用MediQ Fact-Select | Paper显示factuality 89.1% | 需要先跑一个smoke test |

---

## 依赖安装

```bash
pip install groq google-generativeai openai anthropic \
            datasets pyyaml tqdm
```

---

## 还有一个悬而未决的问题

**iMEDQA的"ground truth"字段：** MediQ原始数据是MCQ格式（A/B/C/D选项），ground truth是选项字母。我们的outcome judge需要判断诊断是否正确，这需要把选项字母映射回具体的诊断描述。

loader.py里的`ground_truth_answer`直接存了选项字母（如"B"）和options字典。Judge prompt里要把这个转换为实际文字。**需要你确认一下你们clone的MediQ repo里，iMEDQA的数据字段具体长什么样**，把一个sample的原始JSON格式发给我看一下，我可以帮你把loader写得更准确。
