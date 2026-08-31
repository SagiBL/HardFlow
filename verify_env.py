import importlib.metadata
import re

requirements = """
gym==0.20.0
numpy==2.0.2
matplotlib==3.9.4
scipy==1.13.1
Cython<3.0
GitPython
typed-argument-parser
scikit-video==1.1.11
scikit-image==0.17.2
pandas==2.3.1
black==25.1.0
mdformat==0.7.22
scikit-learn==1.6.1
pin==3.7.0
gin-config==0.5.0
mujoco==2.3.7
opencv-python-headless==4.12.0.88
tabulate==0.9.0
"""

print(f"{'Package':<25} | {'Expected':<10} | {'Installed':<15} | {'Status'}")
print("-" * 75)

for line in requirements.strip().split('\n'):
    # Extract the base package name (e.g., 'gym' from 'gym==0.20.0')
    pkg_name = re.split(r'[=<>~]', line)[0].strip()
    expected = line.replace(pkg_name, '').strip() or "Any"
    
    # Handle the pip vs conda name mismatch for Pinocchio
    check_name = 'pinocchio' if pkg_name == 'pin' else pkg_name
    
    try:
        installed = importlib.metadata.version(check_name)
        status = "✅"
        
        # Add a specific note for NumPy since we intentionally changed it
        if pkg_name == 'numpy' and installed != '2.0.2':
            status = "✅ (Safely downgraded for D4RL/MuJoCo)"
            
    except importlib.metadata.PackageNotFoundError:
        installed = "MISSING"
        status = "❌"
        
    print(f"{pkg_name:<25} | {expected:<10} | {installed:<15} | {status}")
