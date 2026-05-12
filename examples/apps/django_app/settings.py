"""Django settings for the hello world app."""

DEBUG = True
SECRET_KEY = "flyte-django-hello-world-secret-key"
ROOT_URLCONF = "urls"
ALLOWED_HOSTS = ["*"]
INSTALLED_APPS = [
    "django.contrib.contenttypes",
]
MIDDLEWARE = []
