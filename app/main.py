import sys
import os
import sqlite3
import re  

# 终极防线：动态获取当前项目的绝对根目录（E:\github\hetero-multimodal-service）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

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
from openai import OpenAI

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

    # ---------------- 🌟 找回丢失的灵性：预加载你的原生 data.xlsx ----------------
    excel_path = os.path.join(BASE_DIR, "data_file", "data.xlsx")
    if os.path.exists(excel_path):
        try:
            # sheet_name="Sheet1" 指定你的工作表，dtype=str 锁死纯字符读取
            ml_models["ground_truth_df"] = pd.read_excel(excel_path, sheet_name="Sheet1", dtype=str)
            logger.info("📊 [真值表] 成功从 data_file/data.xlsx [Sheet1] 加载黄金标准数据库入内存！")
        except Exception as excel_error:
            logger.error(f"❌ [真值表] Excel文件读取大崩溃！原因: {str(excel_error)}")
    else:
        logger.warning(f"⚠️ [真值表] 未在指定路径找到真实的 data.xlsx 文件: {excel_path}")

    # ---------------- 🌟 SQLite 数据库持久化表初始化 ----------------
    db_path = os.path.join(BASE_DIR, "data_file", "gamut_production.db")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS strawberry_eval_records (
            strawberry_id TEXT PRIMARY KEY,        -- 草莓唯一编号 (主键，防重复)
            file_name TEXT,                       -- 上传的点云文件名
            predicted_grade TEXT,                 -- 神经网络预测形状级别
            predicted_weight REAL,                -- 神经网络预测重量
            true_grade TEXT,                      -- 表格记录的真实形状
            true_weight REAL,                     -- 表格记录的真实重量
            absolute_error REAL,                  -- 系统计算出的绝对误差
            llm_diagnostic_report TEXT            -- 大模型现场手写的专家诊断报告
        );
        """)
        conn.commit()
        conn.close()
        ml_models["db_path"] = db_path
        logger.info("🗄️ [SQLite 数据库] 工业级持久化质检大表初始化/连接成功！")
    except Exception as db_init_error:
        logger.error(f"❌ [SQLite 数据库] 初始化失败: {str(db_init_error)}")

    ml_models["yolo_2d"] = "2D YOLOv8 实例"
    yield
    ml_models.clear()

app = FastAPI(title="2D/3D 异构视觉多任务联合感知网关", lifespan=lifespan)

@app.post("/predict/3d", summary="3D 真实点云分类与真值比对诊断通道")
async def predict_3d(
    file: UploadFile = File(description="请上传真实的草莓 .npy/.ply 点云矩阵文件")
):
    # 1. 检查文件后缀
    allowed_formats = config["models"]["pointcloud_3d"]["supported_formats"]
    file_ext = os.path.splitext(file.filename)[1]
    if file_ext not in allowed_formats:
        raise HTTPException(status_code=400, detail=f"只接收 {allowed_formats} 格式的文件！")
    
    logger.info(f"🧊 [3D 通道] 接收到点云文件: {file.filename}")
    
    try:
        # ==================== 🌟 核心对齐：去零无缝对齐查表机制 ====================
        strawberry_id = "Unknown"
        true_shape = "Unknown"
        true_weight = "Unknown"
        
        # 精准咬住 strawberry 后面的数字
        id_match = re.search(r'strawberry(\d+)', file.filename)
        if id_match:
            raw_id = id_match.group(1)  # 抓出来可能是 "001"
            strawberry_id = raw_id      # 保持原样作为返回 ID
            
            # 强行转成整数去掉前导零，把 "001" 变成 1
            search_int_id = int(raw_id) 
            logger.info(f"🔍 [真值检索] 提取编号: [{raw_id}] -> 统一转换整数 ID 进行查表: [{search_int_id}]")
            
            if "ground_truth_df" in ml_models:
                df = ml_models["ground_truth_df"]
                id_column_name = df.columns[0] # 获取表格第一列的列名
                
                # 核心：把表格的第一列也全部强行转成数字，防止文本/数字类型不一致导致脱靶
                matched_row = df[pd.to_numeric(df[id_column_name], errors='coerce') == search_int_id]
                
                if not matched_row.empty:
                    # 🌟 修正：抛弃数字索引，直接用 Excel 的真实列名（表头）抓取，彻底解决读反的问题！
                    try:
                        # 假设你的 Excel 重量列名叫 "true_weight"，形状列名叫 "true_shape"
                        # 如果你的 Excel 真实列名是中文（如"真实重量"），直接把引号里换成中文即可！
                        true_weight = float(matched_row["true_weight"].iloc[0])
                        true_shape = str(matched_row["true_shape"].iloc[0])
                    except KeyError:
                        # 防御性备用方案：如果列名对不上，自动用更稳妥的动态位置读取
                        true_weight = float(matched_row.iloc[0, 1])
                        true_shape = str(matched_row.iloc[0, 2])
                        
                    logger.info(f"🎯 [真值精准命中] 列名对齐成功！形状: {true_shape}, 重量: {true_weight}g")
        # =======================================================================

        # 2. 读入文件并解析为 NumPy 空间坐标矩阵
        file_bytes = await file.read()
        if file.filename.endswith(".npy"):
            raw_points = np.load(io.BytesIO(file_bytes))
        else:
            temp_file_path = f"temp_{file.filename}"
            with open(temp_file_path, "wb") as f:
                f.write(file_bytes)
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(temp_file_path)
                raw_points = np.asarray(pcd.points)
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        # 3. 变长点云分辨率自动化对齐
        target_num_points = config["models"]["pointcloud_3d"]["num_points"]
        current_num_points = raw_points.shape[0]
        if current_num_points >= target_num_points:
            idx = np.random.choice(current_num_points, target_num_points, replace=False)
            processed_points = raw_points[idx, :]
        else:
            idx = np.random.choice(current_num_points, target_num_points, replace=True)
            processed_points = raw_points[idx, :]
            
        # 4. 🧠 驱动真实的论文 DGCNN 网络进行前向传播
        device = ml_models["device"]
        input_tensor = torch.tensor(processed_points).float().unsqueeze(0).to(device)
        model = ml_models["pointcloud_3d"]
        
        with torch.no_grad():
            cls_logits, weight_pred, _, _ = model(input_tensor)
            predicted_class_idx = cls_logits.argmax(dim=1).item()
            grade_mapping = {0: "Cone_Shape_Grade_A", 1: "Wedge_Shape_Grade_B"}
            final_grade = grade_mapping.get(predicted_class_idx, "Unknown")
            final_weight = float(weight_pred.item())
            
        # 5. 计算算法与真值的绝对误差
        weight_error = "N/A"
        if isinstance(true_weight, (int, float)):
            weight_error = round(abs(final_weight - true_weight), 3)

        # 6. 升级大模型 Prompt，进行真值比对误差诊断
        llm_cfg = config["llm"]
        client = OpenAI(api_key=llm_cfg["api_key"], base_url=llm_cfg["api_base"])
        
        user_prompt = f"""
        【GAMUT 自动化检测诊断请求】
        草莓样本唯一编号: {strawberry_id}
        
        【3D神经网络感知预测值】
        预测等级: {final_grade}
        预测重量: {final_weight:.2f}g
        
        【实验室人工测量黄金标准（真值）】
        真实等级: {true_shape}
        真实重量: {true_weight}g
        双方重量绝对误差: {weight_error}g
        
        请作为‘AI算法评估与作物栽培学交叉专家’，对比上述‘预测值’与‘真实值’，在 180 字内给出一份极简的技术诊断：
        1. 评估算法看准了吗？如果分类对不上或重量误差较大（例如误差>2g），请从点云稀疏度、噪点干扰或真实果实空心失水的角度，技术性分析算法‘看走眼’的潜在工程原因。
        2. 给流水线质检团队的一句话即时现场建议。
        （废话少说，直奔主题，控制在 3 行内！）
        """
        
        completion = client.chat.completions.create(
            model=llm_cfg["model_name"],
            messages=[
                {"role": "system", "content": "你是一个精通 3D 视觉误差分析和 MLOps 评估的专家，擅长找出算法预测值与真值发生偏差的深层物理原因。"},
                {"role": "user", "content": user_prompt}
            ]
        )
        expert_report = completion.choices[0].message.content

        # ==================== 🌟 动作二：将本次实验的全部战果焊进 SQLite 数据库 ====================
        if "db_path" in ml_models:
            try:
                conn = sqlite3.connect(ml_models["db_path"])
                cursor = conn.cursor()
                
                db_sql = """
                INSERT OR REPLACE INTO strawberry_eval_records 
                (strawberry_id, file_name, predicted_grade, predicted_weight, true_grade, true_weight, absolute_error, llm_diagnostic_report)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """
                
                cursor.execute(db_sql, (
                    strawberry_id, 
                    file.filename, 
                    final_grade, 
                    round(final_weight, 3), 
                    true_shape, 
                    true_weight if isinstance(true_weight, (int, float)) else None, 
                    weight_error if isinstance(weight_error, (int, float)) else None, 
                    expert_report
                ))
                conn.commit()
                conn.close()
                logger.info(f"💾 [数据落盘成功] 样本 {strawberry_id} 的全链路多模态指标已被持久化写入数据库！")
            except Exception as db_insert_error:
                logger.error(f"❌ [数据落盘失败] 原因: {str(db_insert_error)}")

        # 7. 打包大厂标准返回
        return {
            "status": "success",
            "strawberry_id": strawberry_id,
            "processed_file": file.filename,
            "model_outputs": {
                "predicted_grade": final_grade,
                "predicted_weight_grams": round(final_weight, 3)
            },
            "ground_truth_metrics": {
                "true_grade": true_shape,
                "true_weight_grams": true_weight,
                "weight_absolute_error_grams": weight_error,
                "is_classification_correct": "YES" if final_grade.split("_")[0].lower() in true_shape.lower() else "NO"
            },
            "llm_error_analysis_report": expert_report
        }
        
    except Exception as e:
        logger.error(f"💥 [3D 通道] 致命错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"系统内部崩溃: {str(e)}")

# ==================== 🌟 动作三：开辟历史大盘数据 SELECT 查询通道 ====================
@app.get("/history", summary="从 SQLite 数据库捞出历史上所有不合格的质检诊断记录")
async def get_history(min_error: float = 0.0):
    if "db_path" not in ml_models:
        raise HTTPException(status_code=500, detail="数据库未正常连接")
        
    try:
        conn = sqlite3.connect(ml_models["db_path"])
        cursor = conn.cursor()
        
        sql_query = "SELECT * FROM strawberry_eval_records WHERE absolute_error >= ? ORDER BY absolute_error DESC;"
        cursor.execute(sql_query, (min_error,))
        all_logs = cursor.fetchall()
        conn.close()
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)