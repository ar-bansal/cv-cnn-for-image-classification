from setuptools import setup, find_packages

setup(
    name="ml",  
    version="1.0.0",  
    packages=find_packages(where=".", include=["ml", "ml.*"]),  
    install_requires=[
        "torch",  
        "torchmetrics",  
        "pytorch-lightning",  
        "numpy",  
        "pandas",  
        "scikit-learn", 
    ],
    include_package_data=True,  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  
)
