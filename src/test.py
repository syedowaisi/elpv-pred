# Create a test.py file in src directory
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Check what's available in models
import models
print("Models package contents:", dir(models))

# Check cnet_model specifically
from models import cnet_model
print("CNet model contents:", dir(cnet_model))

# Try to access the function
if hasattr(cnet_model, 'build_cnet'):
    print("build_cnet found!")
else:
    print("build_cnet NOT found in cnet_model")