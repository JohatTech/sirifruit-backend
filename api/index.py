# Entrypoint alias for hosting platforms that expect api.index:app
from app.main import app

__all__ = ["app"]
