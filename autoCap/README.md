#Enhancing Yelp Reviews through Data Mining

Our website is meant to provide access to our Documentation and Testing Results.
It is constructed with the Django framework, and deployed using Heroku.

**Libraries included for the website:**

Also available in the requirements.txt file

* dj-database-url==0.4.0
* Django==1.9.2
* gunicorn==19.4.5
* psycopg2==2.6.1
* whitenoise==2.0.6

##Website Structure

**Home**

Our homepage displays information about the Yelp Dataset chalenge, and our motivation for the project.

**Reports**

In this section our Abstract is displayed. Links to the project's documentation can be found at the bottom of the page

**Results**

This page displays the pictures from the Yelp dataset.
Clicking on one will enhance it and display further information.
It includes:
* **Original Caption:** The caption that was originally attached to the picture.
* **Predicted Captions:** The captions that resulted in our testing.
* **Original Label:** The label that had originally been given by Yelp to the picture. Labels for restuarants are defined as : inside, outside, food, drink and menu.
* **Predicted Label:** The label that we attached to the photo through our testing.

##Project Structure

The structure follows a Heroku prebuilt Django template. 

0. **autoCap:** Folder containing files for site. 
0. `Procfile:` Needed for Heroku deployment. Specifies activity site should awaken to.
0. `manage.py:` Manages the project's settings.
0. `requirements.txt:` Needed for Heroku deployment. Specifies which libraries are needed in website.
0. `results_84_62.db:` SQLite database of our testing results.

##Website

Site can be found at https://auto-captioning.herokuapp.com