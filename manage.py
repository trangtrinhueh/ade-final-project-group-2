import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forecast_api import create_app

app = create_app()

if __name__ == '__main__':
    print("=" * 50)
    print("  Dashboard chạy tại: http://localhost:5000")
    print("  Ctrl+C để dừng")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000)
