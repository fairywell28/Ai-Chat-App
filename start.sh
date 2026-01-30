#!/bin/bash

# 激活虚拟环境
source venv/bin/activate

# 启动应用
if [ "$ENVIRONMENT" = "production" ]; then
    gunicorn -c gnucorn_conf.py app.main:app
else
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
fi
