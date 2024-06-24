import os
from copy import deepcopy
from typing import Any, Dict, Optional, Type, Union, Callable, TypedDict, Literal
from abc import ABC, abstractmethod
import logging
from dotenv import load_dotenv
import yaml

# Import model classes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms import Ollama
from langchain_anthropic import ChatAnthropic

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_CONFIG_PATH = "config/model-config.yaml"

DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOP_K = None
DEFAULT_TOP_P = None
DEFAULT_TYPICAL_P = 0.8
DEFAULT_OUTPUT_FORMAT = None
DEFAULT_CTX = 4096
DEFAULT_REPEAT_LAST_N = 64
DEFAULT_REPETITION_PENALTY = 1.03
DEFAULT_TFS_Z = 1
DEFAULT_MAX_RETRIES = 6
DEFAULT_SAFETY_SETTINGS = None
DEFAULT_TIMEOUT = None
DEFAULT_NUM_PREDICT = 128
DEFAULT_RAW = True
DEFAULT_SYSTEM = None
DEFAULT_TEMPLATE = None
DEFAULT_FORMAT = "text-generation"


class ModelParams(TypedDict):
    model: str
    temperature: float
    max_tokens: int
    top_k: int
    top_p: float
    typical_p: float
    output_format: str
    ctx: int
    repeat_last_n: int
    repetition_penalty: float
    tfs_z: float
    max_retries: int
    safety_settings: dict
    timeout: int
    num_predict: int
    raw: bool
    system: str
    template: str
    #model_kwargs: dict

class ModelConfig(TypedDict):
    model_class: Type[Any]
    params: ModelParams

class ModelProvider(ABC):
    @abstractmethod
    def create_model(self, **kwargs):
        pass

class GoogleModels(ModelProvider):
    def create_model(self, **kwargs):
        return ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
            temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
            max_output_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            max_retries=kwargs.get("max_retries", DEFAULT_MAX_RETRIES),
            top_p=kwargs.get("top_p", DEFAULT_TOP_P),
            top_k=kwargs.get("top_k", DEFAULT_TOP_K),
            safety_settings=kwargs.get("safety_settings", DEFAULT_SAFETY_SETTINGS),
        )

class OpenAiModels(ModelProvider):
    def create_model(self, **kwargs):
        return ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_organization=os.getenv("OPENAI_ORGANIZATION"),
            temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
        )
    
class AnthropicModels(ModelProvider):
    def create_model(self, **kwargs):
        return ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
            temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            top_p=kwargs.get("top_p", DEFAULT_TOP_P),
            top_k=kwargs.get("top_k", DEFAULT_TOP_K),
        )
    
class GroqModels(ModelProvider):
    def create_model(self, **kwargs):
        return ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_api_base=os.getenv("GROQ_API_BASE"),
            temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            response_format=kwargs.get("output_format", DEFAULT_OUTPUT_FORMAT),
        )

class OllamaModels(ModelProvider):
    def create_model(self, **kwargs):
        return Ollama(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            num_ctx=kwargs.get("ctx", DEFAULT_CTX),
            format=kwargs.get("output_format", DEFAULT_OUTPUT_FORMAT),
            top_p=kwargs.get("top_p", DEFAULT_TOP_P),
            top_k=kwargs.get("top_k", DEFAULT_TOP_K),
            repeat_last_n=kwargs.get("repeat_last_n", DEFAULT_REPEAT_LAST_N),
            repetition_penalty=kwargs.get("repetition_penalty", DEFAULT_REPETITION_PENALTY),
            tfs_z=kwargs.get("tfs_z", DEFAULT_TFS_Z),
            max_retries=kwargs.get("max_retries", DEFAULT_MAX_RETRIES),
            num_predict=kwargs.get("num_predict", DEFAULT_NUM_PREDICT),
            raw=kwargs.get("raw", DEFAULT_RAW),
            system=kwargs.get("system", DEFAULT_SYSTEM),
            template=kwargs.get("template", DEFAULT_TEMPLATE),
        )
    
class HuggingFaceModels(ModelProvider):
    def create_model(self, **kwargs):
        return HuggingFaceEndpoint(
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            #check_docs=os.getenv("HUGGINGFACE_ENDPOINT_URL"),
            max_new_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
            top_p=kwargs.get("top_p", DEFAULT_TOP_P),
            top_k=kwargs.get("top_k", DEFAULT_TOP_K),
            typical_p=kwargs("typical_p", DEFAULT_TYPICAL_P),
            task=kwargs.get("output_format", DEFAULT_OUTPUT_FORMAT),
            repetition_penalty=kwargs.get("repetition_penalty", DEFAULT_REPETITION_PENALTY),
        )

class ModelFactory:
    def __init__(self, config_path: str = MODEL_CONFIG_PATH):
        self.providers: Dict[str, ModelProvider] = {
            "google": GoogleModels(),
            "openai": OpenAiModels(),
            "anthropic": AnthropicModels(),
            "groq": GroqModels(),
            "ollama": OllamaModels(),
            "huggingface": HuggingFaceModels(),
        }

        self.model_config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict[str, ModelConfig]:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def get_model(self, model: str, **kwargs: Any) -> Any:
        """
        Create and return a model instance based on the specified model name and parameters.

        Args:
            model (str): The name of the model to create.
            **kwargs: Additional parameters for model configuration.

        Returns:
            Any: An instance of the requested model.

        Raises:
            ValueError: If the specified model is not supported.
        """
        try:
            model_config = self.model_config[model]
            provider = self.providers[model_config['provider']]
            model_params = deepcopy(model_config['params'])
            model_params.update(kwargs)

            return provider.create_model(**model_params)
        except KeyError:
            logger.error(f"Unsupported model: {model}")
            raise ValueError(f"Unsupported model: {model}")
        except Exception as e:
            logger.exception(f"Error creating model {model}: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        factory = ModelFactory()
        llm = factory.get_model("gpt-4", temperature=0.5, max_tokens=2048)
        # Use the model...
    except ValueError as e:
        logger.error(str(e))
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")