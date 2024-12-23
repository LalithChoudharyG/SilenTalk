from django.shortcuts import render
import re
from django.shortcuts import render, redirect
from django.contrib.auth import logout, login, authenticate
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.contrib import messages

def detection_view(request):
    return render(request, 'detection.html')

User = get_user_model()


def home(request):
    if request.method == 'POST':
        # Handle contact form submission
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        message = request.POST.get('message')

        if not (first_name and last_name and email and subject and message):
            messages.error(request, "All fields are required.")
        else:
            # Construct the full message
            full_message = f"Message from {first_name} {last_name} ({email}):\n\n{message}"

            # Send the email
            send_mail(
                subject=subject,
                message=full_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                # Replace with the support email address
                recipient_list=[settings.DEFAULT_FROM_EMAIL],
                fail_silently=False,
            )
            messages.success(
                request, "Your message has been sent successfully!")

    if request.user.is_authenticated:
        return render(request, 'index.html')
    else:
        return render(request, 'home.html')


def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, "Invalid username or password")

    return render(request, 'login.html')


def signup_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password']
        # password2 = request.POST['password2']

        if len(username) < 3:
            messages.error(
                request, "Username must be at least 3 characters long")
            return redirect('signup')

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            messages.error(request, "Invalid email address")
            return redirect('signup')

        if len(password1) < 8:
            messages.error(
                request, "Password must be at least 8 characters long")
            return redirect('signup')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken")
            return redirect('signup')

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already taken")
            return redirect('signup')

        user = User.objects.create_user(
            username=username, email=email, password=password1)
        user.save()
        messages.success(request, "New User Created")
        return redirect('signup')

    return render(request, 'signup.html')


def logout_view(request):
    logout(request)
    return redirect('home')
