from django.db import models
from django.contrib.auth.models import User


class Detection(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    filename = models.CharField(max_length=255)
    label = models.CharField(max_length=10)
    confidence = models.FloatField()
    fake_probability = models.FloatField()
    visualization_path = models.CharField(max_length=500, null=True, blank=True)
    detailed_metrics = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.filename} - {self.label}"
