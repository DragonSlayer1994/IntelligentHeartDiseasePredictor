from django.conf.urls import url
from . import views

urlpatterns = [

    url(r'^$', views.Prediction.hdPrediction, name='hdPrediction'),

    url(r'^random/$', views.Prediction.forest, name='random'),
]
