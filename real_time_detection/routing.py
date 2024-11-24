from django.urls import path
from detection import consumers

websocket_urlpatterns = [
    path("ws/detect/", consumers.DetectionConsumer.as_asgi()),
]
