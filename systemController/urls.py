from django.conf.urls import url
from . import views

urlpatterns = [

    url(r'^$', views.systemController.systemController, name='systemcontroller'),

    url(r'^hd$', views.systemController.hdpage, name='hd'),

    url(r'^chd$', views.systemController.chdpage, name='chd'),

    url(r'^hdpage/tainhd/$', views.systemController.trainhd, name='trainhd'),

    url(r'^chd/trainchd/$', views.systemController.trainchd, name='trainchd'),
]