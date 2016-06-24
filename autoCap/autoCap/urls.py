# These are the url patterns per page
#This was adapted from Simeon Franklin on August 13, 2015 to Udemy blog here:
#https://blog.udemy.com/django-tutorial-getting-started-with-django/


from django.conf.urls import include, url
from django.contrib import admin

from views import index, tests, report, nonelabel,suggest


urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url(r'^$', index),
    url(r'^tests.html', tests),
    url(r'^report.html', report),
    url(r'^nonelabel.html', nonelabel),
    url(r'^reviewSuggest.html', suggest),
]
