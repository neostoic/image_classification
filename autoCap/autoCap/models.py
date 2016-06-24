"""
    Copyright 2005-2016 Django Software Foundation and individual contributors.
    Django is a registered trademark of the Django Software Foundation.

    This is an auto-generated Django model module.
    Models are the models found in the database.

"""

from __future__ import unicode_literals

from django.db import models


class Images(models.Model):
    photo_id = models.TextField(primary_key=True, blank=True)
    yelp_id = models.TextField(blank=True, null=True)
    business_id = models.TextField(blank=True, null=True)
    file_path = models.TextField(blank=True, null=True)
    file_name = models.TextField(blank=True, null=True)
    split = models.TextField(blank=True, null=True)
    label = models.TextField(blank=True, null=True)
    predicted_label = models.TextField(blank=True, null=True)
    caption = models.TextField(blank=True, null=True)
    predicted_caption_1 = models.TextField(blank=True, null=True)
    caption_confidence_1 = models.TextField(blank=True, null=True)
    predicted_caption_2 = models.TextField(blank=True, null=True)
    caption_confidence_2 = models.TextField(blank=True, null=True)
    predicted_caption_3 = models.TextField(blank=True, null=True)
    caption_confidence_3 = models.TextField(blank=True, null=True)
    predicted_caption_4 = models.TextField(blank=True, null=True)
    caption_confidence_4 = models.TextField(blank=True, null=True)
    predicted_caption_5 = models.TextField(blank=True, null=True)
    caption_confidence_5 = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'images'

class Reviews(models.Model):
    review_id = models.TextField(blank=True, primary_key=True, serialize=False)
    business_id = models.TextField(blank=True, null=True)
    review_text = models.TextField(blank=True, null=True)
    top_words = models.TextField(blank=True, null=True)
    suggested_image1 = models.TextField(blank=True, null=True)
    suggested_image2 = models.TextField(blank=True, null=True)
    suggested_image3 = models.TextField(blank=True, null=True)
    suggested_image4 = models.TextField(blank=True, null=True)
    suggested_image5 = models.TextField(blank=True, null=True)
    topic = models.TextField(blank=True, null = True)

    class Meta:
        managed = False
        db_table = 'reviews'
