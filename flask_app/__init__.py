from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # 配置你的应用
    app.config['SECRET_KEY'] = 'your_secret_key'  # 你可以修改为更安全的密钥
    
    # 注册蓝图（views.py）
    from .routes.views import main
    app.register_blueprint(main)
    
    # 这里还可以导入其他功能模块或初始化扩展
    
    return app
