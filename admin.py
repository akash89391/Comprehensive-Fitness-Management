from django.contrib import admin
from .models import Contact
from .models import Profile
from .models import Strength
# Register your models here.

admin.site.register(Contact)
admin.site.register(Profile)
admin.site.register(Strength)