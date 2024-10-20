import locale

from django.contrib import messages
from django.db.models import Count
from django.db.models.functions import TruncMonth
from django.shortcuts import render, redirect, get_object_or_404
from scipy import stats

from ts_app.models import User, Department, Request, Category, Subcategory, Request

from ts_app.forms import UserForm, DepartmentForm, CategoryForm, SubcategoryForm, RequestForm

from django.shortcuts import render
from .models import Request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64


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


# Представление для отображения списка заявок и формы добавления
def request_list(request):
    requests = Request.objects.all()
    return render(request, 'request_list.html', {'requests': requests})


# Представление для подачи заявки
def request_create(request):
    if request.method == 'POST':
        form = RequestForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Ваша заявка успешно отправлена!')
            return redirect('request_creator')
    else:
        form = RequestForm()
    return render(request, 'request_creator.html', {'form': form})


def plot_histograms(request):
    requests = Request.objects.all()
    locale.setlocale(locale.LC_TIME, 'Russian_Russia.1251')
    date_counts = Request.objects.annotate(month=TruncMonth('creating_date')).values('month').annotate(
        total_requests=Count('id')).order_by('month')

    data = {
        'month': [item['month'] for item in date_counts],
        'total_requests': [item['total_requests'] for item in date_counts]
    }
    df = pd.DataFrame(data)

    # Построение графика распределения
    df['month'] = pd.to_datetime(df['month']).dt.strftime('%B')
    plt.figure(figsize=(10, 6))
    sns.barplot(x='month', y='total_requests', data=df, palette='Reds_d')
    plt.title('Количество заявок по месяцам')
    plt.xlabel('Месяц')
    plt.ylabel('Количество заявок')
    plt.xticks(rotation=90)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Построение Q-Q графика
    values = df['total_requests']
    plt.figure(figsize=(8, 6))
    stats.probplot(values, dist="norm", plot=plt)
    plt.title('Q-Q график для количества заявок по месяцам')
    plt.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    qq_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Построение таблицы кросс-табуляции
    requests_with_executor = Request.objects.exclude(id_executor__isnull=True).filter(id_status=4)

    data_corr = {
        'executor_id': [req.id_executor.id for req in requests_with_executor],
        'executor_name': [req.id_executor.surename for req in requests_with_executor],
        'subcategory_id': [req.id_subcategory.id for req in requests_with_executor],
        'subcategory_name': [req.id_subcategory.name for req in requests_with_executor]
    }

    df = pd.DataFrame(data_corr)
    cross_tab = pd.crosstab(df['executor_id'], df['subcategory_id'])
    executor_map = df[['executor_id', 'executor_name']].drop_duplicates().set_index('executor_id')[
        'executor_name'].to_dict()
    subcategory_map = df[['subcategory_id', 'subcategory_name']].drop_duplicates().set_index('subcategory_id')[
        'subcategory_name'].to_dict()

    cross_tab.index = cross_tab.index.map(executor_map)
    cross_tab.columns = cross_tab.columns.map(subcategory_map)

    vmin_value = cross_tab.values.min()
    vmax_value = cross_tab.values.max()

    plt.figure(figsize=(12, 8))
    sns.heatmap(cross_tab, annot=True, cmap='Reds', linewidths=0.5, vmin=vmin_value, vmax=vmax_value)

    plt.title('Матрица исполнителей по ПО')
    plt.xlabel('ПО')
    plt.ylabel('Исполнитель')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    matrix_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return render(request, 'histogram.html',
                  {'hist_image': image_base64, 'qq_image': qq_image, 'matrix_image': matrix_image})
