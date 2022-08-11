from setuptools import setup, find_packages

setup(
    name='dash_data_viewer',
    python_requires='>=3.10',
    version='1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/TimChild/dash_data_viewer',
    license='MIT',
    author='Tim Child',
    author_email='timjchild@gmail.com',
    description='Dash Viewer for Dats (Folk lab UBC)',
    install_requires=[
        'dat_analysis>=3.0.0',
        'dash>=2.0',
        'plotly',
        'pandas',
        'dash-extensions<=0.0.65',
        'dash-labs',
        'dash-bootstrap-components',
        'dacite',
        'kaleido',
        'filelock',
    ]
)
