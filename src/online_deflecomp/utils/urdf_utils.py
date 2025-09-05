import os

def require_urdf(urdf_path: str = 'simple6r.urdf') -> str:
    abs_path = os.path.abspath(urdf_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f'URDF not found: {abs_path}')
    return abs_path
