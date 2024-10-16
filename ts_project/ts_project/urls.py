from django.contrib import admin
from django.urls import path

from ts_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('users/', views.user_list, name='user_list'),
    path('users/delete/<int:user_id>/', views.delete_user, name='delete_user'),
    path('departments/', views.department_list, name='department_list'),
    path('departments/delete/<int:department_id>/', views.delete_department, name='delete_department'),
    path('categories/', views.category_list, name='category_list'),
    path('categories/delete/<int:category_id>/', views.delete_category, name='delete_category'),
    path('subcategories/', views.subcategory_list, name='subcategory_list'),
    path('subcategories/delete/<int:subcategory_id>/', views.delete_subcategory, name='delete_subcategory'),
    path('requests/', views.request_list, name='request_list'),
    path('requests/create', views.request_create, name='request_creator'),
    path('histogram/', views.plot_histograms, name='plot_histograms'),
    # path('requests/delete/<int:request_id>/', views.delete_request, name='delete_request'),
]
