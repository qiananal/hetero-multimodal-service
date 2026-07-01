import sys
import os
import sqlite3
import time
import aiosqlite
from openai import AsyncOpenAI
import re  



# 终极防线：动态获取当前项目的绝对根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# 🚀 【解耦第一步】：引入我们刚刚做好的专属智能体服务特种兵
from app.agent_service import GamutAgentService

import asyncio
import logging
import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
import yaml
import torch
import numpy as np
import pandas as pd  

# 引入你 model 文件夹里的真实 network 类
from model.model_ablation_no_residual import DGCNN_Ablation_NoResidual
from app.celery_worker import async_heavy_strawberry_task

# 配置日志
log_path = os.path.join(BASE_DIR, "logs", "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")]
)
logger = logging.getLogger("GAMUT-Production-Gateway")

# 加载配置文件
config_yaml_path = os.path.join(BASE_DIR, "config.yaml")
with open(config_yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config["models"]["pointcloud_3d"]["weight_path"] = os.path.join(BASE_DIR, config["models"]["pointcloud_3d"]["weight_path"])

# 全局内存储物柜
ml_models = {}

# 🚀 规范化：全局只初始化唯一一个异步大模型客户端，接口内部绝不二次初始化
llm_cfg = config["llm"]
client = AsyncOpenAI(api_key=llm_cfg["api_key"], base_url=llm_cfg["api_base"])

# 🚀 【解耦第二步】：全局单例初始化智能体大脑服务，main.py 绝不缝合多余逻辑！
agent_service = GamutAgentService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 [Lifespan] 服务器启动，开始装载真实神经网络模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ml_models["device"] = device
    
    # ---------------- 3D 通道模型初始化 ----------------
    gamut_model = DGCNN_Ablation_NoResidual(k=32, dropout=0.2)
    logger.info(f"💾 [3D GAMUT] 正在载入真实权重文件...")
    try:
        checkpoint = torch.load(config["models"]["pointcloud_3d"]["weight_path"], map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            gamut_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            gamut_model.load_state_dict(checkpoint)
        gamut_model.to(device).eval()
        ml_models["pointcloud_3d"] = gamut_model
        logger.info("✅ [3D GAMUT] 论文模型已成功常驻显存！")
    except Exception as e:
        logger.error(f"❌ [3D GAMUT] 权重载入失败: {str(e)}")
        ml_models["pointcloud_3d"] = gamut_model.to(device).eval()

    # ---------------- 📊 预加载你的原生 data.xlsx ----------------
    excel_path = os.path.join(BASE_DIR, "data_file", "data.xlsx")
    if os.path.exists(excel_path):
        try:
            ml_models["ground_truth_df"] = await asyncio.to_thread(pd.read_excel, excel_path, sheet_name="Sheet1", dtype=str)
            logger.info("📊 [真值表] 成功从 data_file/data.xlsx [Sheet1] 加载黄金标准数据库入内存！")
        except Exception as excel_error:
            logger.error(f"❌ [真值表] Excel文件读取大崩溃！原因: {str(excel_error)}")
    else:
        logger.warning(f"⚠️ [真值表] 未在指定路径找到真实的 data.xlsx 文件: {excel_path}")

    # ---------------- 🗄️ SQLite 数据库持久化表初始化 ----------------
    db_path = os.path.join(BASE_DIR, "data_file", "gamut_production.db")
    ml_models["db_path"] = db_path
    try:
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("PRAGMA busy_timeout = 20000;")
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS strawberry_eval_records (
                strawberry_id TEXT PRIMARY KEY,
                file_name TEXT,
                predicted_grade TEXT,
                predicted_weight REAL,
                true_grade TEXT,
                true_weight REAL,
                absolute_error REAL,
                llm_diagnostic_report TEXT
            );
            """)
            await conn.commit()
        logger.info("🗄️ [SQLite 数据库] 工业级持久化质检大表初始化/连接成功！")
    except Exception as db_init_error:
        logger.error(f"❌ [SQLite 数据库] 初始化失败: {str(db_init_error)}")

    ml_models["yolo_2d"] = "2D YOLOv8 实例"
    yield
    ml_models.clear()

app = FastAPI(title="2D/3D 异构视觉多任务联合感知网关", lifespan=lifespan)

@app.post("/predict/3d", summary="3D 真实点云分类与真值比对诊断通道")
async def predict_3d(file: UploadFile = File(description="请上传真实的草莓 .npy/.ply 点云矩阵文件")):
    allowed_formats = config["models"]["pointcloud_3d"]["supported_formats"]
    file_ext = os.path.splitext(file.filename)[1]
    if file_ext not in allowed_formats:
        raise HTTPException(status_code=400, detail=f"只接收 {allowed_formats} 格式的文件！")
    
    logger.info(f"🧊 [前线网关] 接收到高频相机点云冲击: {file.filename}")
    try:
        strawberry_id = "Unknown"
        id_match = re.search(r'strawberry(\d+)', file.filename)
        if id_match:
            strawberry_id = id_match.group(1)

        file_bytes = await file.read()
        async_result = async_heavy_strawberry_task.delay(strawberry_id, file.filename, file_bytes)
        
        return {
            "status": "QUEUED",
            "msg": "🍓 草莓质检任务已成功攻入后台 Redis 缓冲舱，流量削峰安全着陆！",
            "task_id": async_result.id,
            "strawberry_id": strawberry_id
        }
    except Exception as e:
        logger.error(f"💥 [前线网关] 派发异步任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"系统内部崩溃: {str(e)}")

@app.get("/history", summary="从 SQLite 数据库捞出历史上所有不合格的质检诊断记录")
async def get_history(min_error: float = 0.0):
    if "db_path" not in ml_models:
        raise HTTPException(status_code=500, detail="数据库未正常连接")
    try:
        async with aiosqlite.connect(ml_models["db_path"]) as conn:
            sql_query = "SELECT * FROM strawberry_eval_records WHERE absolute_error >= ? ORDER BY absolute_error DESC;"
            cursor = await conn.execute(sql_query, (min_error,))
            all_logs = await cursor.fetchall()
        
        history_list = []
        for row in all_logs:
            history_list.append({
                "strawberry_id": row[0],
                "file_name": row[1],
                "predicted_metrics": {"grade": row[2], "weight_grams": row[3]},
                "ground_truth_metrics": {"true_grade": row[4], "true_weight_grams": row[5]},
                "absolute_error_grams": row[6],
                "llm_expert_report": row[7]
            })
        return {
            "status": "success",
            "total_records_found": len(history_list),
            "filtering_criteria": f"absolute_error >= {min_error}g",
            "data": history_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取历史数据库失败: {str(e)}")

from celery.result import AsyncResult
from app.celery_worker import celery_app 

@app.get("/predict/status/{task_id}", summary="MLOps 专属任务状态机追踪网关")
async def get_task_status(task_id: str):
    try:
        async_result = AsyncResult(task_id, app=celery_app)
        current_state = async_result.state
        response_data = {
            "task_id": task_id,
            "status": current_state,
            "msg": "⏳ 正在同步排队或计算中..."
        }
        if current_state == "SUCCESS":
            response_data["msg"] = "✅ 后台算力大获全胜，战果已成功焊入 SQLite 数据库！"
            response_data["result"] = async_result.result 
        elif current_state == "FAILURE":
            response_data["msg"] = "💥 幕后计算大崩溃，可能由于文件解析错误或大模型超时！"
            response_data["error_details"] = str(async_result.info) 
        return response_data
    except Exception as e:
        logger.error(f"❌ [状态网关] 查验状态失败, 单号: {task_id}, 原因: {str(e)}")
        raise HTTPException(status_code=500, detail=f"状态机同步失败: {str(e)}")

# =========================================================================
# 🌊 异构核心合龙区：大模型 SSE 网关 + 智能体解耦异步触发
# =========================================================================
from fastapi.responses import StreamingResponse
import json

@app.get("/predict/llm_stream/{strawberry_id}", summary="AIGC 专属 SSE Token 级流式网关")
async def stream_strawberry_diagnostic(strawberry_id: str, pred_shape: str, pred_weight: float):
    
    user_prompt = f"""
    【GAMUT 自动化检测流式诊断请求】
    草莓样本编号: {strawberry_id}
    视觉网络预测几何形状: {pred_shape}
    视觉网络预测绝对重量: {pred_weight:.2f}g
    请在 180 字内给出极简技术诊断和一句话车间现场操作建议。
    """

    async def llm_token_generator():
        try:
            response = await client.chat.completions.create(
                model=config["llm"]["model_name"],
                messages=[
                    {"role": "system", "content": "你是一个精通 3D 视觉误差分析和 MLOps 评估的专家。"},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True  
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"❌ [SSE流网关] 大模型流式断裂, 原因: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            
        finally:
            # ─── 🌟 终极高光时刻：接力棒交接！ ───
            # 当大模型把字全蹦完了，或者连接意外意外挂断，只要退出管道，立刻在这里触发智能体！
            logger.info(f"🕵️‍♂️ [解耦控制流] SSE 传输结束。正在凌空触发后台智能体进行车间现场决策...")
            
            # 由于智能体包含模型调用，属于计算密集型/IO阻塞型任务，为了防止卡死网关长连接，
            # 我们动用 asyncio.to_thread 把它一脚踹进独立的后台线程池里偷偷跑！
            # 这里硬编码模拟拿到实验室真值 24.50g（也可以去读持久化后的 SQLite 捞出真正真值）
            asyncio.create_task(
                asyncio.to_thread(
                    agent_service.run_decision_flow,
                    strawberry_id=strawberry_id,
                    pred_shape=pred_shape,
                    pred_weight=pred_weight,
                    true_weight=24.50
                )
            )

    return StreamingResponse(
        llm_token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",   
            "Connection": "keep-alive",    
            "X-Accel-Buffering": "no"      
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)