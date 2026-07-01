# 1. 基础环境：直接从云端拉取一个做过 AI 优化的官方 Python 3.10 环境
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 2. 设置工作目录：在集装箱内部自动建一个叫 /app 的文件夹，并走进去
WORKDIR /app

# 3. 复制依赖清单：把本地的依赖列表先单独复制到集装箱里
COPY requirements.txt .

# 4. 安装依赖：在集装箱内运行 pip 安装，换清华源加速，并清理缓存给集装箱瘦身
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 5. 代码拷贝：把本地当前文件夹下的所有核心代码，全部复制进集装箱
COPY . .

# 6. 接口开放：告诉外部，这个集装箱运行起来后，会占用 8000 端口
EXPOSE 8000

# 7. 启动命令：当集装箱被拉起运行（Container）时，默认执行的主入口脚本
CMD ["python", "main.py"]