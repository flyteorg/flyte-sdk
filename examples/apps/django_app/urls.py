"""URL configuration for the Django hello world app."""

from django.http import HttpResponse, JsonResponse
from django.urls import path


def hello_world(request):
    """Return a simple hello world HTML page."""
    return HttpResponse(
        "<h1>Hello, World!</h1>"
        "<p>Welcome to Django on Flyte!</p>"
        "<ul>"
        "<li><a href='/health'>Health Check</a></li>"
        "<li><a href='/api/info'>API Info</a></li>"
        "</ul>"
    )


def health(request):
    """Health check endpoint."""
    return JsonResponse({"status": "healthy", "framework": "django"})


def api_info(request):
    """API info endpoint."""
    return JsonResponse({
        "app": "Django Hello World",
        "version": "1.0.0",
        "message": "This Django app is served via Flyte AppEnvironment",
    })


urlpatterns = [
    path("", hello_world, name="hello"),
    path("health", health, name="health"),
    path("api/info", api_info, name="api_info"),
]
