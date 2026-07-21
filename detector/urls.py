from django.urls import path
from .views import (
    home, detect_video, signup_view, login_view, logout_view, 
    dashboard, record_detail, delete_record, delete_all_records,
    custom_admin_dashboard, delete_user_view
)

urlpatterns = [
    path("", home, name="home"),
    path("detect/", detect_video, name="detect_video"),
    path("signup/", signup_view, name="signup"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
    path("dashboard/", dashboard, name="dashboard"),
    path("dashboard/<int:pk>/", record_detail, name="record_detail"),
    path("dashboard/<int:pk>/delete/", delete_record, name="delete_record"),
    path("dashboard/delete_all/", delete_all_records, name="delete_all_records"),
    path("admin-dashboard/", custom_admin_dashboard, name="custom_admin"),
    path("admin-dashboard/delete/user/<int:user_id>/", delete_user_view, name="delete_user"),
]