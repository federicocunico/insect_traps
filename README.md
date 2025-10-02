## Requirements:
- Nodejs (>22.x) and npm (>10.x)
- Python (>3.x)

Install requirements with:
```bash
pip install -r backend/requirements.txt
```

## (1) Setup frontend:
```bash
cd frontend
npm install
npm run build
```

## (2) Setup backend:
```bash
cd ../backend
# Create a virtual environment (optional but recommended)
python -m venv venv
# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

## (3) Run the application:
```bash
cd backend && python server.py
```
The application will be accessible at `http://localhost:8000`.