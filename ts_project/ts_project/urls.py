from django.contrib import admin
from django.urls import path

from ts_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('users/', views.user_list, name='user_list'),
    path('users/delete/<int:user_id>/', views.delete_user, name='delete_user'),  # Удаление пользователя

]
