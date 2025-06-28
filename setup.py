from setuptools import setup, find_packages

setup(
    name="Heart Disease Prediction Model",  
    version="0.1.0",
    author="Chandan Chaudhari", 
    author_email="chaudhari.chandan22@gmail.com",  
    description="Predict whether a person is likely to have heart disease based on medical data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/chandanc5525/EnE_HearDiesease_ClassificationModel", 
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "mlflow",
        "PyYAML",
        "joblib",
        "matplotlib",  
        "tqdm",        
        "seaborn"      
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
