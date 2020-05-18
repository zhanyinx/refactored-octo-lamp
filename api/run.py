"""Run the app."""

from .octolamp import app

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
