from setuptools import setup, find_packages

setup(
    name="clinical_triage_env",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openenv-core>=0.2.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.6.0",
        "openai>=1.30.0",
        "python-dotenv>=1.0.0",
        "httpx>=0.27.0",
    ],
    entry_points={
        "console_scripts": [
            "server=clinical_triage_env.server.app:main",
        ],
    },
)
