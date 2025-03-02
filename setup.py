from setuptools import find_packages, setup

package_name = 'hand_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alextseng',
    maintainer_email='tsung.yuan.tseng@rwth-aachen.de',
    description='A package that uses mediapipe to track hand gesture.',
    license='GNU General Public License v3.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_tracking = hand_tracking.hand_tracking:main'
        ],
    },
)
