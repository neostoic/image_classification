# These are the url patterns per page

from django.conf.urls import include, url
from django.contrib import admin

from views import index, tests, report


urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url(r'^$', index),
    url(r'^tests.html', tests),
    url(r'^report.html', report),
]
