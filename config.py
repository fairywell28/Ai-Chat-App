import os
from dotenv import load_dotenv

load_dotenv()  # 读取环境变量.env文件

class Config:
    """基础环境配置"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_app.db")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))


class DevelopmentConfig(Config):
    """开发环境配置"""


class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    OPENAI_BASE_URL = "https://api.lingyaai.cn/v1/"


class ProductionConfig(Config):
    """生产环境配置"""


config = {
    'base': Config,
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,

    'default': DevelopmentConfig
}

def get_config():
    env = os.getenv('FLASK_ENV', 'default')
    config_class = config.get(env)
    return config_class()
