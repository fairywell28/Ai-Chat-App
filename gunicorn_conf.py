import multiprocessing

# 服务器绑定
bind = "0.0.0.0:8000"

# 工作进程数
workers = multiprocessing.cpu_count() * 2 + 1

# 工作模式
worker_class = "uvicorn.workers.UvicornWorker"

# 日志配置
accesslog = "-"
errorlog = "-"
loglevel = "info"

# 超时设置
timeout = 120
keepalive = 5
