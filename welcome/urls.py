from django.conf.urls import url
from . import views
from django.contrib.auth import views as auth_views
from django.contrib import admin
from django.views.generic.base import TemplateView

from django.conf.urls import url, include
from django.contrib.auth import views as auth_views

urlpatterns = [

    url(r'^h$', TemplateView.as_view(template_name='home.html'), name='home'),
    url(r'^login/$', auth_views.login, {'template_name': 'login.html'}, name='login'),
    url(r'^logout/$', auth_views.logout, {'template_name': 'welcome.html'}, name='logout'),
    url(r'^$', views.Welcome.welcome, name='welcome'),
    url(r'^signup/$', views.Welcome.signup, name='signup'),
    url(r'^activate/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20})/$',
        views.Welcome.activate, name='activate'),

]
