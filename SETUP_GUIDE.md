# Deepfake Detection Demo Setup Guide

This guide will walk you through setting up and running the Deepfake Detection project on your local machine.

## Prerequisites

1. **Python**: Ensure you have Python installed (preferably version 3.9 to 3.11). You can download it from [python.org](https://www.python.org/downloads/). When installing on Windows, make sure to check the box that says **"Add Python to PATH"**.

## Installation Steps

1. **Extract the Project**:
   Extract the provided zip file to a folder on your computer.

2. **Open Terminal / Command Prompt**:
   Open your command prompt or terminal and navigate to the extracted project folder (the folder containing `manage.py` and `requirements.txt`).

3. **Create a Virtual Environment (Recommended)**:
   It's best practice to install dependencies in an isolated environment so they don't conflict with other Python projects.
   ```bash
   python -m venv venv
   ```

4. **Activate the Virtual Environment**:
   - **On Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **On macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

5. **Install Dependencies**:
   With your virtual environment activated, run the following command to install all necessary packages:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This might take a few minutes because it downloads some large machine learning packages like PyTorch, Meta's TIMM, and OpenCV.*

6. **Apply Database Migrations**:
   Set up the local SQLite database by running:
   ```bash
   python manage.py migrate
   ```

7. **Create a Superuser (Optional)**:
   If you want to log into the admin/dashboard panel as an administrator:
   ```bash
   python manage.py createsuperuser
   ```
   Follow the prompts to set a username and password.

## Running the Application

1. **Start the Django Development Server**:
   ```bash
   python manage.py runserver
   ```

2. **Access the Website**:
   Open any web browser and go to: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## Important Notes
- Ensure your internet connection is active when running the application for the first time, as the ML code might download pre-trained weights automatically.
- To stop the server later, press `CTRL-C` in your terminal. When you're completely done, you can type `deactivate` to exit the virtual environment.
