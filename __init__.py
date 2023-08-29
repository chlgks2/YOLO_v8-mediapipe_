from flask import Flask
from flask_mail import Mail, Message

mail = Mail()

def create_app():
    app = Flask(__name__)

    app.config['MAIL_SERVER'] = '주소'  # Gmail SMTP 서버 주소
    app.config['MAIL_PORT'] = 123   # TLS/STARTTLS용 포트
    app.config['MAIL_USE_TLS'] = True  # TLS 필요 (TLS/STARTTLS 사용)
    app.config['MAIL_USE_SSL'] = False  # SSL 사용하지 않음 (TLS/STARTTLS를 사용하므로)
    app.config['MAIL_USERNAME'] = '유저네임쓰셈요'  
    app.config['MAIL_PASSWORD'] = '비번쓰셈요'  # 2023 08 23

    mail.init_app(app)
    

    from .views.obj_detect_views import result_blueprint
    from .views import obj_detect_views, quiz_views
    app.register_blueprint(obj_detect_views.bp)
    app.register_blueprint(quiz_views.bp)

    app.register_blueprint(result_blueprint)


#     with app.app_context():
#             obj_detect_views.mail.init_app(app)
            
    return app