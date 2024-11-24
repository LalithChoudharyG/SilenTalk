import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import real_time_detection.routing

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "real_time_detection.settings")

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            real_time_detection.routing.websocket_urlpatterns
        )
    ),
})
