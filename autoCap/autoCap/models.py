# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey has `on_delete` set to the desired behavior.
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
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
