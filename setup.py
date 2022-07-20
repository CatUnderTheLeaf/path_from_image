from setuptools import setup

package_name = 'path_from_image'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'sympy',
        'cv2'
    ],
    zip_safe=True,
    maintainer='Stacy',
    maintainer_email='catundertheleaf@gmail.com',
    description='A ROS package, which gets real world lane coordinates with only image, camera calibration and position info.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_warper = path_from_image.image_warper:main',
            'lane_area_drawer = path_from_image.lane_area_drawer:main'
        ],
    },
)
