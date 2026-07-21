from django.contrib import admin
from .models import Detection

@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ('filename', 'user', 'label', 'confidence', 'created_at')
    list_filter = ('label', 'created_at')
    search_fields = ('filename', 'user__username')
