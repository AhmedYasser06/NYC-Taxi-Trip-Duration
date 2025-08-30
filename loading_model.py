import os
import joblib

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def abs_path(*paths):
    return os.path.join(BASE_DIR, *paths)

def load_model(file_path):
    try:
        return joblib.load(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")

if __name__ == "__main__":
    print(f"Project root: {BASE_DIR}")
