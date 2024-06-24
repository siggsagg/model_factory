# ModelFactory

ModelFactory is a versatile tool for AI projects that simplifies the integration and management of multiple models from open and closed source providers like OpenAI, Google, and Ollama. It is framework-independent, making it ideal for projects that require frequent model changes and testing or using various models simultaneously.

# Table of Contents

1. [Introduction and Purpose](#introduction-and-purpose)
2. [Features](#features)
3. [Benefits](#benefits)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Supported Parameters](#supported-parameters)
6. [Contributing](#contributing)
7. [Support and Community](#support-and-community)
8. [Future Plans](#future-plans)
9. [Contact](#contact)
10. [License](#license)

## Introduction and Purpose

The main purpose of ModelFactory is to make it easier to test and implement several models into your AI projects. Different models have different strengths and weaknesses that are suited for various tasks. System requirements, token limits, and parameter support vary across models, making it challenging to manage them manually. ModelFactory abstracts these complexities, translating parameters and handling model-specific requirements automatically.

The idea for ModelFactory originated from my challenges with token limits on my CrewAI projects, which prevented continuous operation without interruption. To overcome this, I needed to use different models, which introduced a significant learning curve. ModelFactory was born from the need to switch models effortlessly and assign them to agents seamlessly.

## Features

- **Easily Switch Models**: Seamlessly switch between different models without changing your code.
- **Parameter Management**: Use ModelFactory parameters to tweak your models without remembering provider-specific parameters.
- **Automatic Setup**: Models are set up with the correct templates automatically.
- **Broad Compatibility**: Implement ModelFactory into most applications that support LangChain.
- **Easy Install for Local Models**: Just run the [model]-create_model.sh file to download and install the model with the correct model templates for optimal model output performance.

ModelFactory can be integrated into various applications, offering unique flexibility and ease of use.

## Benefits

- **Simplified Integration**: Quickly switch and test different models in your projects.
- **Avoid Token Limits**: Use multiple models to bypass token request limits from a single provider.
- **System Flexibility**: Accommodate different system requirements by easily downgrading or or upgrading by switching models.
- **Parameter Translation**: ModelFactory handles parameter translation, so you don’t have to remember the specific parameters for each model provider.
- **Improved Efficiency**: Automatically set up models with the correct templates, saving time and increase output performance.

## Installation

Clone the repository into your project directory and install the dependencies using Poetry or the requirements.txt file. If you are integrating ModelFactory into an existing project, you can alternatively copy the dependencies from the pyproject.toml or requirements.txt files into your existing project's dependency management files and install them from there.

### Windows

```bash
git clone https://github.com/username/model_factory.git
cd model_factory
poetry install
# or
pip install -r requirements.txt
```

## Usage

ModelFactory supports most model configurations as parameters:

```python
from model_factory import get_model

factory = ModelFactory()
llm1 = factory.get_model(model="gpt-4o")
llm2 = factory.get_model(model="gpt-4o", temperature=0.2, max_tokens=4096)
llm3 = factory.get_model(model="o-llama3", temperature=1, max_tokens=2048, top_p=0.9, top_k=40, tfs_z=1)
```

If you use Ollama and want to add and download local models from Ollama, you can run the `model_reference-create_model.sh` file before using the model.

### Supported Parameters

| Parameter          | Type     | Optional/Obligatory | Default  | Description                 | Example             |
|--------------------|----------|---------------------|----------|-----------------------------|---------------------|
| model              | str      | Obligatory          | None     | Model name                  | "gpt-4o"            |
| temperature        | float    | Optional            | 0.8      | Sampling temperature        | 0.2                 |
| max_tokens         | int      | Optional            | 4096     | Maximum number of tokens    | 2048                |
| top_k              | int      | Optional            | None     | Top-k sampling              | 40                  |
| top_p              | float    | Optional            | None     | Top-p sampling              | 0.9                 |
| typical_p          | float    | Optional            | None     | Typical-p sampling          | 0.8                 |
| format             | str      | Optional            | None     | Output format               | "text-generation"   |
| repetition_penalty | float    | Optional            | 1.03     | Repetition penalty          | 1.2                 |
| tfs_z              | float    | Optional            | 1        | Tail free sampling          | 1                   |

For more details about the models supported, see the README_MODELS.md file.

## Contributing

Contributions are welcome! If you want to contribute, please send me a private message explaining what you want to contribute, and we'll work out the details in a respectful and collaborative manner.

## Support and Community

Join my Discord community [Agentic AI Automations and Dev](https://discord.gg/MkAJhFe2) or join the [Simplify AI](https://www.skool.com/career-pathway-5625) Skool community where I am an active member and can answer questions.

## Future Plans

- [] Finishing the MODEL_README.md to include all currently supported models.
- [] Automatic model download and installation on any model first time assignment
- [] Ading more models as they are released or requested
- [] Adding support for HuggingFace models

## Contact

For support, inquiries and feature requests, use the Discord channel or the Skool community mentioned above.

## License

This project is licensed under the MIT License.

## Contact

If you have any questions, feel free to reach out through the provided community channels.
