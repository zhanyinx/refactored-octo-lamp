"""Run the app."""

import sys

sys.path.append("../")

from api.octolamp import app

if __name__ == '__main__':
    app.run(host='localhost', port=8889)
