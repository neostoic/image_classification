#autoCap

This folder contains files needed for Django site.

##Structure

The structure follows a Heroku prebuilt Django template. 

0. **migrations:** Folder contains files for database migrations needed by Django. 
0. **static:** Contains static files, including css files. Static photos are stored in external database.
0. **templates:** Contains html files for website pages.
0. `__init__.py:` This file is empty and it is only included as a requirement for all packages.
0. `models.py:` Contains database table models in Django.
0. `settings.py:` Containg setting for Django project, including directories of project files.
0. `urls.py:` Contails url pattern configurations for website pages.
0. `views.py:` Contains queries and handles site requests.
0. `wsgi.py:` Needed for Heroku deployment.

##Website

Site can be found at https://auto-captioning.herokuapp.com