import os
import sys
import yaml
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 🚀 强力双保险拉齐根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# =========================================================================
# ⚙️ 1. 叫醒我们的大模型大脑
# =========================================================================
with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

llm_cfg = config["llm"]
# 绑定我们随身带的两个车间硬件工具按钮
model = ChatOpenAI(
    model=llm_cfg["model_name"],
    api_key=llm_cfg["api_key"],
    base_url=llm_cfg["api_base"],
    temperature=0.1
)

# =========================================================================
# 🛠️ 2. 准备两个车间干活的真实 Python 函数（工具箱）
# =========================================================================
def send_workshop_alert_email(manager_name: str, strawberry_id: str, error_g: float):
    print(f"\n🚨 [邮件组件被AI成功触发] ──> 正在向车间主任 {manager_name} 邮箱发送警报！")
    print(f"🚨 [警报内容]: 草莓 {strawberry_id} 号绝对误差高达 {error_g}g！")
    return "成功！邮件已送达！"

def drive_stepper_motor_reject(strawberry_id: str):
    print(f"\n🦾 [物理硬件被AI成功驱动] ──> 步进电机发出轰鸣！物理挡板已将草莓 {strawberry_id} 号拨进废料桶！")
    return "成功！坏果已剔除！"


# =========================================================================
# 🧱 3. 提示词模具：逼迫大模型不做长篇大论，必须严格“报菜名”返回 JSON 格式
# =========================================================================
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个精通车间流水线控制的 Agent 专家。
    请根据客人的输入做出决策。你必须【严格且仅仅】返回一个 JSON 字典，绝对不允许带任何寒暄废话。
    
    格式规范必须长这样：
    {{
        "need_email": true或false,
        "need_reject": true或false,
        "reason": "你的思考原因"
    }}
    
    【车间守则】：
    1. 如果绝对误差超过 5.0g，need_email 必须为 true。
    2. 如果是严重发霉残次果，need_reject 必须为 true。"""),
    ("human", "{input_data}")
])

# 用 LangChain 最优雅的管道符，把模具和大模型连接在一起
# 这条流水线的意思是：输入填空 ──> 塞给大模型 ──> 吐出干净的字符串
agent_chain = prompt_template | model


# =========================================================================
# 🎬 4. 【核心灵魂】纯手工锻造一个“大厂级智能体执行器大卫兵”
# =========================================================================
def my_custom_agent_executor(raw_input_text: str):
    print("\n" + "⚡"*10 + " GAMUT 智能体开始自我思考 " + "⚡"*10)
    
    # A. 把草莓数据塞进填空题，逼大模型做选择题
    response = agent_chain.invoke({"input_data": raw_input_text})
    ai_speech = response.content # 拿到大模型吐出来的 JSON 字符串
    
    print(f"🤖 [大模型内部决策输出]:\n{ai_speech}\n")
    
    # B. 卫兵在后台用嘴把大模型的文字解析成 Python 字典
    decision = json.loads(ai_speech)
    
    print("🕵️‍♂️ [执行器卫兵开始听名字下厨]...")
    # C. 卫兵开始根据大模型的选择，在原地帮你运行对应的 Python 硬件函数！
    if decision.get("need_email") == True:
        send_workshop_alert_email(manager_name="袁先生", strawberry_id="088", error_g=6.4)
        
    if decision.get("need_reject") == True:
        drive_stepper_motor_reject(strawberry_id="088")
        
    print("\n🎉 " + "🔥"*5 + " 流水线最高决策全盘执行完毕！安全收工！ " + "🔥"*5)


if __name__ == "__main__":
    # 🔬 模拟场景：车间抓到了一个绝对误差高达 6.4g 的发霉残次草莓！
    test_input = "当前检测草莓编号: 088 号 | 绝对误差: 6.40g | 等级: 严重发霉残次果。请决策！"
    
    # 启动我们自己搓的无敌执行器！
    my_custom_agent_executor(test_input)