import os
import sys
import io
import re
import time
import yaml
import torch
import numpy as np
import pandas as pd
from celery import Celery
import asyncio
from openai import AsyncOpenAI
import aiosqlite

# -------------------------------------------------------------------------
# 🚀 强力破壁：动态获取项目【最外层】的绝对根目录
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 🌟 核心双保险：算出 app/ 文件夹的绝对路径
APP_DIR = os.path.join(BASE_DIR, "app")

print("\n" + "="*50)
print(f"🔍 [工兵环境体检] 最外层根目录 BASE_DIR: {BASE_DIR}")
print(f"🔍 [工兵环境体检] 内部核心 app 目录 APP_DIR: {APP_DIR}")
print("="*50 + "\n")

# 🚀 强行轰炸：把根目录和 app 目录全部插队塞进最头部（索引 0）
# 完美模拟并兼容 main.py 在 app/ 目录下运行时的黄金视域环境！
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)  # 👈 让 app/ 目录享受到最高优先级查找

# 🚀 双重保险坐镇大门口，Celery 再执行跨包导入，内鬼被当场清除！
from model.model_ablation_no_residual import DGCNN_Ablation_NoResidual

# 1. 初始化后台中央工兵大队
celery_app = Celery(
    "gamut_tasks",
    broker="redis://127.0.0.1:6379/0",
    backend="redis://127.0.0.1:6379/0"
)

# 2. 预加载基础配置（工兵开机时仅加载一次）
config_yaml_path = os.path.join(BASE_DIR, "config.yaml")
with open(config_yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 3. 初始化大模型异步客户端
llm_cfg = config["llm"]
client = AsyncOpenAI(api_key=llm_cfg["api_key"], base_url=llm_cfg["api_base"])

# 4. 【高薪黑科技】工兵挂机时，直接将真实 3D 神经网络和 Excel 加载入后台进程内存！
print("👷 [Celery 初始化] 正在载入真实 3D 神经网络权重常驻后台显存...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_path = os.path.join(BASE_DIR, config["models"]["pointcloud_3d"]["weight_path"])

gamut_model = DGCNN_Ablation_NoResidual(k=32, dropout=0.2)
try:
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        gamut_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        gamut_model.load_state_dict(checkpoint)
    gamut_model.to(device).eval()
    print("✅ [Celery 初始化] 3D 神经网络模型载入成功，工兵挂机待命！")
except Exception as e:
    print(f"❌ [Celery 初始化] 神经网络载入失败: {str(e)}")

# 预加载 Excel 黄金真值表入后台内存
excel_path = os.path.join(BASE_DIR, "data_file", "data.xlsx")
ground_truth_df = None
if os.path.exists(excel_path):
    ground_truth_df = pd.read_excel(excel_path, sheet_name="Sheet1", dtype=str)
    print("📊 [Celery 初始化] 黄金标准真值表成功加载进后台内存字典！")


# 🚀 5. 铸造终极完全体后台算法工兵任务
@celery_app.task(name="celery_worker.async_heavy_strawberry_task")
def async_heavy_strawberry_task(strawberry_id: str, file_name: str, file_bytes: bytes):
    print(f"\n👷 [工兵流水线] ====> 开始处理草莓流水线任务: ID={strawberry_id}, 文件={file_name}")
    
    try:
        true_shape = "Unknown"
        true_weight = "Unknown"
        
        # ---------------- 动作一：后台内存查表 ----------------
        if ground_truth_df is not None:
            search_int_id = int(strawberry_id)
            id_column_name = ground_truth_df.columns[0]
            matched_row = ground_truth_df[pd.to_numeric(ground_truth_df[id_column_name], errors='coerce') == search_int_id]
            
            if not matched_row.empty:
                try:
                    true_weight = float(matched_row["true_weight"].iloc[0])
                    true_shape = str(matched_row["true_shape"].iloc[0])
                except KeyError:
                    true_weight = float(matched_row.iloc[0, 1])
                    true_shape = str(matched_row.iloc[0, 2])
                print(f"🎯 [工兵查表] 成功咬住实验室真值 -> 形状: {true_shape}, 重量: {true_weight}g")

        # ---------------- 动作二：解析跨进程传过来的二进制字节 ----------------
        if file_name.endswith(".npy"):
            raw_points = np.load(io.BytesIO(file_bytes))
        else:
            temp_file_path = os.path.join(BASE_DIR, "app", f"temp_{file_name}")
            with open(temp_file_path, "wb") as f:
                f.write(file_bytes)
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(temp_file_path)
                raw_points = np.asarray(pcd.points)
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        # 🚀 严格核对：变长点云分辨率自动化对齐（确保 processed_points 在两个分支都名正言顺地定义）
        target_num_points = config["models"]["pointcloud_3d"]["num_points"]
        current_num_points = raw_points.shape[0]
        
        if current_num_points >= target_num_points:
            idx = np.random.choice(current_num_points, target_num_points, replace=False)
            processed_points = raw_points[idx, :]  # 👈 确保这一行严丝合缝地躺在这里
        else:
            idx = np.random.choice(current_num_points, target_num_points, replace=True)
            processed_points = raw_points[idx, :]  # 👈 确保这一行严丝合缝地躺在这里
        # ---------------- 动作三：驱动显卡跑网络前向传播 ----------------
        input_tensor = torch.tensor(processed_points).float().unsqueeze(0).to(device)
        with torch.no_grad():
            cls_logits, weight_pred, _, _ = gamut_model(input_tensor)
            predicted_class_idx = cls_logits.argmax(dim=1).item()
            grade_mapping = {0: "Cone_Shape_Grade_A", 1: "Wedge_Shape_Grade_B"}
            final_grade = grade_mapping.get(predicted_class_idx, "Unknown")
            final_weight = float(weight_pred.item())
            
        weight_error = "N/A"
        if isinstance(true_weight, (int, float)):
            weight_error = round(abs(final_weight - true_weight), 3)

        # ---------------- 动作四：用 asyncio 桥梁强行驱动大模型异步网络调用 ----------------
        user_prompt = f"""
        【GAMUT 自动化检测诊断请求】
        草莓样本唯一编号: {strawberry_id}
        预测等级: {final_grade} | 预测重量: {final_weight:.2f}g
        真实等级: {true_shape} | 真实重量: {true_weight}g
        双方重量绝对误差: {weight_error}g
        请在 180 字内给出极简技术诊断和一句话现场建议。
        """
        
        # 💡 大厂工程规范：由于 Celery 原生不支持 async 函数，我们在这里利用
        # asyncio.get_event_loop().run_until_complete() 强行拉起一个微型事件循环，安全等待大模型
        async def fetch_llm():
            completion = await client.chat.completions.create(
                model=config["llm"]["model_name"],
                messages=[
                    {"role": "system", "content": "你是一个精通 3D 视觉误差分析和 MLOps 评估的专家。"},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return completion.choices[0].message.content
            
        expert_report = asyncio.get_event_loop().run_until_complete(fetch_llm())

        # ---------------- 动作五：用 asyncio 桥梁将战果强幂等焊进 SQLite 数据库 ----------------
        async def save_to_db():
            db_path = os.path.join(BASE_DIR, "data_file", "gamut_production.db")
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute("PRAGMA busy_timeout = 20000;")
                db_sql = """
                INSERT OR REPLACE INTO strawberry_eval_records 
                (strawberry_id, file_name, predicted_grade, predicted_weight, true_grade, true_weight, absolute_error, llm_diagnostic_report)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """
                await conn.execute(db_sql, (
                    strawberry_id, file_name, final_grade, round(final_weight, 3),
                    true_shape, true_weight if isinstance(true_weight, (int, float)) else None,
                    weight_error if isinstance(weight_error, (int, float)) else None, expert_report
                ))
                await conn.commit()
                
        asyncio.get_event_loop().run_until_complete(save_to_db())
        print(f"💾 [工兵流水线] ✅ 草莓 {strawberry_id} 质检大捷！战果已成功焊入物理硬盘！")
        return {"status": "SUCCESS", "strawberry_id": strawberry_id}

    except Exception as e:
        print(f"💥 [工兵流水线] 核心计算大崩溃: {str(e)}")
        return {"status": "FAILED", "error": str(e)}