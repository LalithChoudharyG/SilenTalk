# detection/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('detection/login/', views.login_view, name='login'),
    path('detection/signup/', views.signup_view, name='signup'),
    path('detection/logout/', views.logout_view, name='logout'),
    path('detection/detection/', views.detection_view, name='detection'),
]
