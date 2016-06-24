"""
    Copyright 2005-2016 Django Software Foundation and individual contributors.
    Django is a registered trademark of the Django Software Foundation.

    Views return the request to the page and send database items
    Queries are filtered and requested here along with django classes

    Some of this code was based from Simeon Franklin on August 13, 2015 to Udemy blog here:
    https://blog.udemy.com/django-tutorial-getting-started-with-django/

    Last view uses a database other than the default one
    Paginator is an included class in django that creates pages from the information given by the query
    This was adapted from the django Documentation
    https://docs.djangoproject.com/en/1.9/


"""
import os
from settings import BASE_DIR
from django.shortcuts import render
from .models import Images, Reviews
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.http import HttpResponse
from wsgiref.util import FileWrapper

def index(request):
    return render(request, "index.html")

def report(request):
    return render(request, "report.html")

def tests(request):
    images = Images.objects.all().filter(split='test')
    paginator = Paginator(images, 40) # Show 40 images per page
    page = request.GET.get('page')
    try:
        images = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        images = paginator.page(1)
    except EmptyPage:
        # If page is out of range (e.g. 9999), deliver last page of results.
        images = paginator.page(paginator.num_pages)

    return render(request, "tests.html", {'images': images})

def nonelabel(request):
    images = Images.objects.all().filter(split='test', label='none')
    paginator = Paginator(images, 40) # Show 40 images per page
    page = request.GET.get('page')
    try:
        images = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        images = paginator.page(1)
    except EmptyPage:
        # If page is out of range (e.g. 9999), deliver last page of results.
        images = paginator.page(paginator.num_pages)

    return render(request, "nonelabel.html", {'images': images})

def suggest(request):
    reviews = Reviews.objects.using('reviews').all()
    paginator = Paginator(reviews, 1) # Show 1 images per page
    page = request.GET.get('page')
    try:
        reviews = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        reviews = paginator.page(1)
    except EmptyPage:
        # If page is out of range (e.g. 9999), deliver last page of results.
        reviews = paginator.page(paginator.num_pages)

    return render(request, "reviewSuggest.html", {'reviews': reviews})

