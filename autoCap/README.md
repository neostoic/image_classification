#Enhancing Yelp Reviews through Data Mining

Our website is meant to provide access to our Documentation and Testing Results.
It is constructed with the Django framework, and deployed using Heroku.

**Libraries included for the website:**

Also available in the requirements.txt file

*dj-database-url==0.4.0
*Django==1.9.2
*gunicorn==19.4.5
*psycopg2==2.6.1
*whitenoise==2.0.6

## Home

Our homepage displays information about the Yelp Dataset chalenge, and our motivation for the project.

## Reports

In this section our Abstract is displayed. Links to the project's documentation can be found at the bottom of the page

##Results

This page displays the pictures from the Yelp dataset.
Clicking on one will enhance it and display further information.
It includes:
* **Original Caption:** The caption that was originally attached to the picture.
* **Predicted Captions:** The captions that resulted in our testing.
* **Original Label:** The label that had originally been given by Yelp to the picture. Labels for restuarants are defined as : inside, outside, food, drink and menu.
* **Predicted Label:** The label that we attached to the photo through our testing.

##Website

Site can be found at https://auto-aptioning.herokuapp.com