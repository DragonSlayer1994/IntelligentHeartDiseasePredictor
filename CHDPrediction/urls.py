from django.conf.urls import  url
from . import views

urlpatterns = [


    url(r'^$', views.CoronaryPrediction.CHDPrediction, name='CHDPrediction'),
    url(r'^train$', views.CoronaryPrediction.train, name='train'),
]
