from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404

from ts_app.models import User

from ts_app.forms import UserForm


def user_list(request):
    users = User.objects.all()
    if request.method == 'POST':
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()  # Сохраняем нового пользователя
            return redirect('user_list')  # Перенаправляем на страницу списка пользователей
    else:
        form = UserForm()  # Пустая форма для добавления
    return render(request, 'user_list.html', {'users': users, 'form': form})


# Представление для удаления пользователя
def delete_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    user.delete()  # Удаляем пользователя
    return redirect('user_list')  # Возвращаемся к списку пользователей
