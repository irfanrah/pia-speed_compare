import platform
from setuptools import setup, find_packages

common_requires = [
    'redis==5.2.1',
    'ultralytics==8.3.227',
    'rich==15.0.0',
    'pytz==2024.2',
    'python-dotenv==1.0.1',
    'pika==1.3.2',
    'retry==0.9.2',
    'shapely',
    'boto3',
    'transformers==4.57.3',
    'dill==0.4.0',
    'numba==0.61.2',
    'pydantic==2.10.3',
    'GPUtil',
    'pre-commit',
    'pytest',
    'tensorboard',
    'torchreid==0.2.5',
    'lap==0.5.12',
    'filterpy',
    # 'yolox @ git+https://github.com/noahcao/OC_SORT.git@7d06bffe98b5e57cc696ce56739554483c7e99ca',
]

system = platform.system()
machine = platform.machine()

platform_requires = []

if system == 'Linux' and machine == 'aarch64':
    platform_requires += ['opencv-python']
elif system == 'Linux':
    platform_requires += ['opencv-python==4.10.0.84', 'tensorrt==10.8.0.43']
elif system == 'Windows':
    platform_requires += ['opencv-python==4.10.0.84']
elif system == 'Darwin':
    platform_requires += ['opencv-python==4.10.0.84']

setup(
    name='pia_prod',
    version='2.0.0',
    packages=find_packages(where="packages"),
    package_dir={'': 'packages'},
    install_requires=common_requires + platform_requires,
    python_requires='>=3.10',
)
