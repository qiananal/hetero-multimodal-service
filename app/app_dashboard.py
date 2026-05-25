import sqlite3
import os
import streamlit as st  
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data_file", "gamut_production.db")

st.set_page_config(page_title="GAMUT 工业级质检综合大屏", layout="wide")

st.title("🍓 GAMUT 3D多模态联合感知 ── 工业级实时评测监控大盘")
st.markdown("---")

# 侧边栏：真值手动在线修正 (UPDATE)
st.sidebar.header("🎛️ 车间主控台")
st.sidebar.subheader("✏️ 黄金真值在线校准")
input_id = st.sidebar.text_input("请输入要修正的草莓编号（如 001）", value="")
new_weight = st.sidebar.number_input("请输入修正后的真实重量 (g)", min_value=0.0, value=20.0, step=0.1)

if st.sidebar.button("💾 确认一键校准真值"):
    if input_id and os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            update_sql = """
            UPDATE strawberry_eval_records 
            SET true_weight = ?, 
                absolute_error = round(abs(predicted_weight - ?), 3)
            WHERE strawberry_id = ?;
            """
            cursor.execute(update_sql, (new_weight, new_weight, input_id))
            conn.commit()
            conn.close()
            st.sidebar.success(f"🎉 成功！样本 {input_id} 的数据已物理修正，误差已重算！")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"修正失败: {str(e)}")

if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1️⃣ 高级 SQL 聚合统计：COUNT 和 AVG (对应大厂面试考点)
    cursor.execute("SELECT COUNT(*), AVG(absolute_error) FROM strawberry_eval_records;")
    summary_res = cursor.fetchone()
    total_count = summary_res[0] if summary_res else 0
    avg_mae = round(summary_res[1], 3) if summary_res and summary_res[1] is not None else 0.0
    
    # 画出大厂最爱的巨型指标卡
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="📊 全厂累计检测草莓总数", value=f"{total_count} 颗")
    with col2:
        st.metric(
            label="📉 算法平均绝对误差 (MAE)", 
            value=f"{avg_mae} g", 
            delta="- 优" if avg_mae < 2.0 else "+ 偏差较大", 
            delta_color="normal" if avg_mae < 2.0 else "inverse"
        )
    
    st.markdown("---")

    # 2️⃣ 用 Pandas 捞出台账原始数据
    df = pd.read_sql_query("SELECT * FROM strawberry_eval_records;", conn)
    conn.close()

    if not df.empty:
        # ==================== 🌟 今日高光：算法回归性能双曲线看板 ====================
        st.subheader("📈 GAMUT 算法推理拟合度实时监控（数据科学视图）")
        
        # 为了让折线图能漂亮地画出两条线，我们把大表里最值钱的两列抽出来
        # 并把索引设为草莓编号，这样 X 轴就会自动变成草莓的身份证号
        chart_data = df[["strawberry_id", "predicted_weight", "true_weight"]].copy()
        chart_data.columns = ["草莓编号", "🤖 神经网络预测重量 (g)", "⚖️ 实验室真实重量 (g)"]
        chart_data.set_index("草莓编号", inplace=True)
        
        # 工业界最经典的单行命令：Streamlit 自动帮你把两个维度的趋势画在同一个画板上
        st.line_chart(chart_data, height=350)
        st.markdown("---")
        # =========================================================================

        st.subheader("📋 流水线历史检测总台账（SQL 实时持久化数据）")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("⚠️ 历史数据库目前是空的，请先去 FastAPI 后端跑几次 3D 点云预测！")
else:
    st.error(f"❌ 未找到数据库文件！")