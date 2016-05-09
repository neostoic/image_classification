# This is an auto-generated Django model module.
# Models are the models found in the database.

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
    predicted_caption_2 = models.TextField(blank=True, null=True)
    predicted_caption_3 = models.TextField(blank=True, null=True)
    predicted_caption_4 = models.TextField(blank=True, null=True)
    predicted_caption_5 = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'images'
