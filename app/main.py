# coding: utf-8
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.database import create_tables
from app.api import chat
from config import Config
import os
from dotenv import load_dotenv

# 加载环境变量和 openai_key，临时用，TODO将来用 config 文件替代
load_dotenv()
# 从环境变量中获取 API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 创建数据库表
create_tables()

# 创建FastAPI应用实例
app = FastAPI(
    title="AI智能对话系统",
    description="基于Python和AI大模型的智能对话应用",
    version="1.0.0"
)
#app.config.from_object(Config['development'])

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 配置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 注册路由
app.include_router(chat.router)


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
