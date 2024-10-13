from django.contrib import messages
from django.shortcuts import render, redirect, get_object_or_404

from ts_app.models import User, Department, Request, Category, Subcategory

from ts_app.forms import UserForm, DepartmentForm, CategoryForm, SubcategoryForm


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
    if Request.objects.filter(id_executor=user).exists() or Request.objects.filter(id_user=user).exists():
        messages.error(request, 'Этого пользователя нельзя удалить')
        return redirect('user_list')  # Возвращаемся к списку пользователей
    else:
        messages.success(request, f'Пользователь {user.surename} {user.name} {user.patronymic} был удален')
        user.delete()  # Удаляем пользователя
        return redirect('user_list')  # Возвращаемся к списку пользователей


# Представление для отображения списка отделов и формы добавления
def department_list(request):
    departments = Department.objects.all()

    if request.method == 'POST':
        form = DepartmentForm(request.POST)
        if form.is_valid():
            form.save()  # Сохраняем новый отдел
            return redirect('department_list')  # Перенаправляем на список отделов
    else:
        form = DepartmentForm()

    return render(request, 'department_list.html', {'departments': departments, 'form': form})


# Представление для удаления отдела
def delete_department(request, department_id):
    department = get_object_or_404(Department, id=department_id)
    if User.objects.filter(id_department=department).exists():
        messages.error(request, 'Невозможно удалить отдел, в котором есть сотрудники.')
        return redirect('department_list')
    else:
        department = get_object_or_404(Department, id=department_id)
        department.delete()
        messages.success(request, f'Отдел {department.name} был удален')
        return redirect('department_list')


# Представление для отображения списка категорий и формы добавления
def category_list(request):
    categories = Category.objects.all()

    if request.method == 'POST':
        form = CategoryForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('category_list')
    else:
        form = CategoryForm()

    return render(request, 'category_list.html', {'categories': categories, 'form': form})


# Представление для удаления категорий
def delete_category(request, category_id):
    category = get_object_or_404(Category, id=category_id)
    if Request.objects.filter(id_category=category).exists():
        messages.error(request, 'Невозможно удалить эту категорию')
    else:
        category.delete()
        messages.success(request, f'Категория {category.name} была удалена')
    return redirect('category_list')

# Представление для отображения списка подкатегорий и формы добавления
def subcategory_list(request):
    subcategories = Subcategory.objects.all()
    if request.method == 'POST':
        form = SubcategoryForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('subcategory_list')
    else:
        form = SubcategoryForm()

    return render(request, 'subcategory_list.html', {'subcategories': subcategories, 'form': form})


# Представление для удаления подкатегорий
def delete_subcategory(request, subcategory_id):
    subcategory = get_object_or_404(Subcategory, id=subcategory_id)
    if Request.objects.filter(id_subcategory=subcategory).exists():
        messages.error(request, 'Невозможно удалить эту подкатегорию')
    else:
        subcategory.delete()
        messages.success(request, f'Подкатегория {subcategory.name} была удалена')
    return redirect('subcategory_list')
