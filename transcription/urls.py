from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('transcription/', views.transcription, name='transcription'),
    path('transcription/details/<int:id>', views.details, name='details'),
]
