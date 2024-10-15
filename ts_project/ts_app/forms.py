from django import forms
from .models import User, Department, Category, Subcategory, Request


class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['name', 'surename', 'patronymic', 'email', 'id_department']

        labels = {
            'surename': 'Фамилия',
            'name': 'Имя',
            'patronymic': 'Отчество',
            'email': 'Email',
            'id_department': 'Отдел'
        }

        widgets = {
            'surename': forms.TextInput(attrs={'class': 'small-input'}),
            'name': forms.TextInput(attrs={'class': 'small-input'}),
            'patronymic': forms.TextInput(attrs={'class': 'small-input'}),
            'email': forms.EmailInput(attrs={'class': 'small-input'}),
            'department': forms.Select(attrs={'class': 'small-input'})
        }


class DepartmentForm(forms.ModelForm):
    class Meta:
        model = Department
        fields = ['name']
        labels = {
            'name': 'Название отдела'
        }
        widgets = {
            'name': forms.TextInput(attrs={'class': 'small-input'})
        }


class CategoryForm(forms.ModelForm):
    class Meta:
        model = Category
        fields = ['name']
        labels = {
            'name': 'Название категории'
        }
        widgets = {
            'name': forms.TextInput(attrs={'class': 'small-input'})
        }


class SubcategoryForm(forms.ModelForm):
    class Meta:
        model = Subcategory
        fields = ['name']
        labels = {
            'name': 'Название подкатегории'
        }
        widgets = {
            'name': forms.TextInput(attrs={'class': 'small-input'})
        }


class RequestForm(forms.ModelForm):
    class Meta:
        model = Request

        fields = ['id_user', 'text', 'id_category', 'id_subcategory']

        labels = {
            'id_user': 'ФИО пользователя',
            'text': 'Текст',
            'id_category': 'Категория',
            'id_subcategory': 'Подкатегория',
        }

        widgets = {
            'user': forms.Select(attrs={'class': 'small-input'}),
            'text': forms.Textarea(),
            'category': forms.Select(attrs={'class': 'small-input'}),
            'subcategory': forms.Select(attrs={'class': 'small-input'}),
        }
