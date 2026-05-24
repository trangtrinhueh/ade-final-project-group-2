import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flask import Flask


def create_app():
    _templates = os.path.join(_ROOT, 'templates')
    app = Flask(__name__, template_folder=_templates)

    from .routes import register_routes
    register_routes(app)

    return app
