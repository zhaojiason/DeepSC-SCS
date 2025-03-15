from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # 配置你的应用
    app.config['SECRET_KEY'] = 'your_secret_key'  # 你可以修改为更安全的密钥
    
    # 注册路由
    from .routes.views import index
    app.add_url_rule('/', view_func=index, methods=['GET', 'POST'])
   
    return app