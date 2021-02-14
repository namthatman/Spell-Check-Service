from django.urls import path

from . import views

app_name = 'base'

urlpatterns = [
    # api
    path('textprocess', views.textprocess, name='text_process')
]