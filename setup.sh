#!/bin/bash

echo "Setting up street-aware-app (React)..."
cd street-aware-app
npm install
cd ..

echo "Setting up street-aware-service (FastAPI)..."
cd street-aware-service
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
deactivate
cd ..

echo "Setting up street-aware-scripts (Python tools)..."
cd street-aware-scripts
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
deactivate
cd ..

echo "✅ Setup complete."
