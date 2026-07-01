import os
import yaml
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class GamutAgentService:
    def __init__(self):
        # 1. 叫醒大脑
        with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        llm_cfg = config["llm"]
        self.model = ChatOpenAI(
            model=llm_cfg["model_name"],
            api_key=llm_cfg["api_key"],
            base_url=llm_cfg["api_base"],
            temperature=0.1
        )
        
        # 2. 焊死提示词模具
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个精通流水线控制的 Agent 专家。
            你必须【严格且仅仅】返回一个 JSON 字典，绝对不允许带任何寒暄废话。
            格式规范：
            {{
                "need_email": true或false,
                "need_reject": true或false,
                "reason": "你的思考原因"
            }}
            【车间守则】：
            1. 如果绝对误差超过 5.0g，need_email 必须为 true。
            2. 如果是严重发霉残次果（Grade_C），need_reject 必须为 true。"""),
            ("human", "{input_data}")
        ])
        
        # 3. 组装最核心的极简管道
        self.agent_chain = self.prompt_template | self.model

    # 🛠️ 工具按钮 A：发邮件
    def _send_workshop_alert_email(self, strawberry_id: str, error_g: float):
        print(f"\n🚨 [解耦智能体] ──> 正在向车间主任袁先生邮箱疯狂发送弹窗警报！")
        print(f"🚨 [警报内容]: 发现草莓 {strawberry_id} 号误差高达 {error_g}g！")
        return "邮件送达"

    # 🛠️ 工具按钮 B：步进电机踢果
    def _drive_stepper_motor_reject(self, strawberry_id: str):
        print(f"\n🦾 [解耦智能体] ──> 步进电机发出轰鸣！机械臂已将草莓 {strawberry_id} 号拨进废料桶！")
        return "物理剔除"

    # 🎬 核心对外公开的“点火”大闸门
    def run_decision_flow(self, strawberry_id: str, pred_shape: str, pred_weight: float, true_weight: float):
        """
        供外部调用的最高决策接口。输入当前的质检数据，自动思考并驱动车间硬件。
        """
        error_g = round(abs(pred_weight - true_weight), 3)
        
        # 填空并塞给大模型
        input_text = f"草莓编号: {strawberry_id} | 视觉预测重量: {pred_weight}g | 真实重量: {true_weight}g (绝对误差: {error_g}g) | 几何等级: {pred_shape}"
        response = self.agent_chain.invoke({"input_data": input_text})
        
        try:
            # 听大模型“报菜名”，后台 Python 开始“下厨”
            decision = json.loads(response.content)
            
            actions_taken = []
            if decision.get("need_email") == True:
                res = self._send_workshop_alert_email(strawberry_id, error_g)
                actions_taken.append(res)
                
            if decision.get("need_reject") == True:
                res = self._drive_stepper_motor_reject(strawberry_id)
                actions_taken.append(res)
                
            return {
                "decision": decision,
                "actions_taken": actions_taken,
                "status": "COMPLETED"
            }
        except Exception as e:
            return {"status": "ERROR", "detail": f"智能体大脑断流: {str(e)}"}