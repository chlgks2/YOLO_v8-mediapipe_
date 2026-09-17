# --- 자격증명 분리 : 하드코딩된 키를 .env 로 옮겼습니다 ---
import os
from pathlib import Path


def _load_dotenv():
    here = Path(__file__).resolve().parent
    for d in [here, *here.parents][:4]:
        f = d / ".env"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return


_load_dotenv()

from flask import Flask
from flask_mail import Mail, Message

mail = Mail()

def create_app():
    app = Flask(__name__)

    app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
    app.config['MAIL_USE_TLS'] = True  # TLS 필요 (TLS/STARTTLS 사용)
    app.config['MAIL_USE_SSL'] = False  # SSL 사용하지 않음 (TLS/STARTTLS를 사용하므로)
    app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', '')
    app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '')

    mail.init_app(app)
    

    from .views.obj_detect_views import result_blueprint
    from .views import obj_detect_views, quiz_views
    app.register_blueprint(obj_detect_views.bp)
    app.register_blueprint(quiz_views.bp)

    app.register_blueprint(result_blueprint)


#     with app.app_context():
#             obj_detect_views.mail.init_app(app)
            
    return app