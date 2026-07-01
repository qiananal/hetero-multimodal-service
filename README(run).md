# 启动说明

这个项目包含三部分：
- FastAPI 后端：负责接收点云文件、调度任务
- Celery Worker：负责后台推理与数据库写入
- Streamlit 前端：提供可视化大屏

## 1. 环境准备

### Python 环境
```powershell
cd E:\github\hetero-multimodal-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Redis
这个项目的 Celery 依赖 Redis，因此至少需要先启动 Redis。

#### 方式 A：Docker 启动 Redis（推荐）
```powershell
docker run -d --name gamut-redis -p 6379:6379 redis:7.0-alpine
```

#### 方式 B：本机已安装 Redis
```powershell
redis-server
```

---

## 2. 本地启动方式（推荐开发）

建议开 3 个终端，分别启动下面 3 个进程。

### 终端 1：启动 Celery Worker
```powershell
cd E:\github\hetero-multimodal-service
.\.venv\Scripts\Activate.ps1
celery -A app.celery_worker.celery_app worker --loglevel=info -P solo
```

### 终端 2：启动 FastAPI 服务
```powershell
cd E:\github\hetero-multimodal-service
.\.venv\Scripts\Activate.ps1
python app/main.py
```

### 终端 3：启动 Streamlit 前端
```powershell
cd E:\github\hetero-multimodal-service
.\.venv\Scripts\Activate.ps1
python -m streamlit run app/app_dashboard.py
```

启动成功后：
- FastAPI： http://127.0.0.1:8000
- Streamlit： http://localhost:8501

---

## 3. Docker 启动方式

仓库里已经提供了 Dockerfile，可用于构建后端镜像。

### 3.1 启动 Redis 容器
```powershell
docker run -d --name gamut-redis -p 6379:6379 redis:7.0-alpine
```

### 3.2 构建后端镜像
```powershell
docker build -t gamut-service .
```

### 3.3 运行后端容器
```powershell
docker run --rm -it -p 8000:8000 --name gamut-service gamut-service
```

> 说明：当前仓库的 Docker 方案主要用于启动后端服务，Celery Worker 和 Streamlit 仍建议在本地终端中运行。

---

## 4. 常见问题

### Redis 连不上
检查是否已经启动：
```powershell
docker ps
```

### 权重文件找不到
请确认以下文件存在：
- weights/GAMUT_fold0.pth
- weights/yolov8n.pt

### LLM 调用失败
检查 [config.yaml](config.yaml) 中的 API Key 和模型配置是否正确。