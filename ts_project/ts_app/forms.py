from django import forms
from .models import User


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
