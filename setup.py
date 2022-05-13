import setuptools

setuptools.setup(
    name="thermocore",
    version="0.0.1",
    packages=["thermocore", "thermocore.geometry", "thermocore.io"],
    install_requires=["numpy", "scipy"],
    python_requires=">=3.7",
)
