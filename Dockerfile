# 使用官方 Python 镜像作为基础镜像，选择合适的版本
FROM python:3.10-slim

# 设置容器中工作的目录
WORKDIR /app

# 将当前目录下的所有文件复制到容器的工作目录中
COPY . /app

# 安装项目所需的依赖
# 先复制 requirements.txt 文件，避免每次构建都重新安装依赖
#COPY requirements.txt /app/
RUN pip install -e .

# 如果有其他依赖文件或环境变量可以在这里添加

# 默认命令可以设置为运行你主程序的命令
#CMD ["python", "your_main_script.py"]
