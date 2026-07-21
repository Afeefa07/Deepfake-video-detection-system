import os
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required

from django.contrib.auth.models import User
from django.db.models import Q

from .models import Detection
from .predict import predict_video

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def home(request):
    return render(request, "index.html")


def signup_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("dashboard")
    else:
        form = UserCreationForm()
    return render(request, "signup.html", {"form": form})


def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            if user.is_superuser:
                return redirect("custom_admin")
            return redirect("dashboard")
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {"form": form})


def logout_view(request):
    logout(request)
    return redirect("home")


@login_required
def dashboard(request):
    history = Detection.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "dashboard.html", {"history": history})

@login_required
def record_detail(request, pk):
    from django.shortcuts import get_object_or_404
    item = get_object_or_404(Detection, pk=pk, user=request.user)
    return render(request, "detail.html", {"item": item})

@login_required
def delete_record(request, pk):
    from django.shortcuts import get_object_or_404, redirect
    if request.method == "POST":
        item = get_object_or_404(Detection, pk=pk, user=request.user)
        item.delete()
    return redirect("dashboard")

@login_required
def delete_all_records(request):
    from django.shortcuts import redirect
    if request.method == "POST":
        Detection.objects.filter(user=request.user).delete()
    return redirect("dashboard")


@csrf_exempt
def detect_video(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    if "video" not in request.FILES:
        return JsonResponse({"error": "No video uploaded"}, status=400)

    video = request.FILES["video"]
    path = os.path.join(UPLOAD_DIR, video.name)

    with open(path, "wb+") as f:
        for chunk in video.chunks():
            f.write(chunk)

    try:
        result = predict_video(path)
        
        user = request.user if request.user.is_authenticated else None
        
        Detection.objects.create(
            user=user,
            filename=video.name,
            label=result.get("label", "UNKNOWN"),
            confidence=result.get("confidence", 0.0),
            fake_probability=result.get("fake_probability", 0.0),
            visualization_path=result.get("visualization_path", ""),
            detailed_metrics=result
        )
        
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

@login_required
def custom_admin_dashboard(request):
    if not request.user.is_superuser:
        return redirect("dashboard")
        
    users = User.objects.all().order_by('-date_joined')
    recent_detections = Detection.objects.select_related('user').order_by('-created_at')[:50]
    total_users = users.count()
    total_videos = Detection.objects.count()
    
    # Calculate fakes (cases where label is not REAL)
    total_fakes = Detection.objects.filter(~Q(label='REAL')).count()
    
    context = {
        'users': users,
        'recent_detections': recent_detections,
        'total_users': total_users,
        'total_videos': total_videos,
        'total_fakes': total_fakes,
    }
    return render(request, "custom_admin_dashboard.html", context)

@login_required
def delete_user_view(request, user_id):
    if not request.user.is_superuser:
        return redirect("dashboard")
        
    if request.method == "POST":
        from django.shortcuts import get_object_or_404
        user_to_delete = get_object_or_404(User, id=user_id)
        # Prevent self-deletion
        if user_to_delete != request.user:
            user_to_delete.delete()
            
    return redirect("custom_admin")