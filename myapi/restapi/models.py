from django.conf import settings
from django.db import models
from django.conf import settings

# Create your models here.

class Hero(models.Model):
    name = models.CharField(max_length=60)
    alias = models.CharField(max_length=60)
    picture = models.ImageField(upload_to=settings.MEDIA_ROOT, blank =True)

    def __str__(self):
        return self.name
