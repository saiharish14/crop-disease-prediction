import webbrowser
import threading
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
PORT = int(os.environ.get('PORT', 5000))
DEBUG = os.environ.get('FLASK_DEBUG', '1').lower() in ('1', 'true', 'yes', 'on')
OPEN_BROWSER = os.environ.get('OPEN_BROWSER', '1').lower() in ('1', 'true', 'yes', 'on')

def open_browser():
    webbrowser.open_new(f'http://127.0.0.1:{PORT}/')

def should_open_browser():
    if os.environ.get('PORT') or not OPEN_BROWSER:
        return False

    if DEBUG:
        return os.environ.get('WERKZEUG_RUN_MAIN') == 'true'

    return True

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model or scaler file not found. Please run the notebook to generate 'model.pkl' and 'scaler.pkl'.")
    else:
        from app import app
        if should_open_browser():
            threading.Timer(1.5, open_browser).start()

        app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
