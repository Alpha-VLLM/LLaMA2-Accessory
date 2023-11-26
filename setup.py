import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LLaMA2-Accessory",
    version="0.0.1",
    author="Alpha-VLLM",
    description="An Open-source Toolkit for LLM Development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Alpha-VLLM/LLaMA2-Accessory",
    packages=setuptools.find_packages(),
    include_package_data=True,
)
