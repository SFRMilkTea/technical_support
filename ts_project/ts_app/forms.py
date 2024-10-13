from django import forms
from .models import User, Department, Category


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
