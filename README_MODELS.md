# Model Specs and Information

[Some text]

## Table of Contents

[Table of contents]

## Term Explanations

Below you'll find detailed explanations of key terms and parameters used in the ModelFactory. Understanding these terms will help you effectively configure and utilize the models to suit your specific needs. Each term is crucial for tailoring the model's performance, output quality, and resource utilization. Below, you'll find descriptions of parameters, context window, quantization, and model reference, which will guide you in optimizing the model's behavior for various applications.

### Parameters

The ModelFactory supports a range of parameters that allow users to customize the behavior of their models. These parameters influence the model's output, performance, and behavior in various ways. Below is an explanation of each parameter and its effect:

| Parameter          | Type     | Optional/Obligatory | Default   | Description                 | Example             | Provider Support                |
|:-------------------|:---------|:--------------------|----------:|:----------------------------|:--------------------|:--------------------------------|
| model              | str      | Required            | None      | Model name                  | "gpt-4o"            | All                             |
| temperature        | float    | Optional            | 0.8       | Sampling temperature        | 0.2                 | All                             |
| max_tokens         | int      | Optional            | 4096      | Maximum output tokens       | 2048                | Google, OpenAI Anthropic, Groq  |
| top_k              | int      | Optional            | None      | Top-k sampling              | 40                  | Google, Ollama                  |
| top_p              | float    | Optional            | None      | Top-p sampling              | 0.9                 | Google, Groq, Ollama            |
| typical_p          | float    | Optional            | None      | Typical-p sampling          | 0.8                 | HuggingFace                     |
| output_format      | str      | Optional            | None      | Output format               | "text-generation"   | Ollama                          |
| ctx                | int      | Optional            | None      | Context Window              | 128000              | Ollama                          |
| repeat_last_n      | int      | Optional            | 64        | Penalty look back distance  | 0                   | Ollama                          |
| repetition_penalty | float    | Optional            | 1.03      | Repetition penalty          | 1.2                 | Ollama                          |
| tfs_z              | float    | Optional            | 1         | Tail free sampling          | 1                   | Ollama                          |
| max_retries        | int      | Optional            | 6         | Max retries for completion  | 2                   | Google, OpenAI, Anthropic, Groq |
| safety_settings    | dict     | Optional            | None      | Load Google Safety Settings | *See below table    | Google                          |
| timeout            | int      | Optional            | None      | Timeout for request stream  | 3600                | Google, Anthropic               |
| num_predict        | int      | Optional            | 128       | Maximum tokens to predict   | -2                  | Ollama                          |
| raw                | bool     | Optional            | None      | Model keep alive instruct   | *See below table    | Ollama                          |
| system             | str      | Optional            | None      | Override ModelFile Config   | *See below table    | Ollama                          |
| template           | str      | Optional            | None      | Override ModelFile Config   | *See below table    | Ollama                          |
| model_kwargs       | dict     | Optional            | None      | Holds unspecified params    | **FUTURE FEATURE**  | None                            |

safety_settings:
raw: <https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately>
system: <https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.ollama.ChatOllama.html>
template: <https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.ollama.ChatOllama.html>

### Parameter Details

#### model

- **Type**: `str`
- **Example**: `"gpt-4o"` or `"o-phi3-3b-128k-q6"`

**Description**:
Specifies the name of the model to be used. This is a mandatory parameter that identifies which model should be utilized for generating outputs. The model string is specific to the ModelFactory, and only supports model references that are supported by ModelFactory. Go to the [Model Reference](#model-reference) section for detailed information on constructing and using model reference strings.

#### temperature

- **Type**: `float`
- **Default**: `0.8`
- **Example**: `0.2`

**Description**:
Temperature is a hyperparameter used to control the randomness of the model's output in language generation tasks. By adjusting the temperature, you can influence the creativity and variability of the generated text.

- **Lower Temperature Values**: When the temperature is set to a lower value (e.g., 0.2), the model's output becomes more deterministic and focused. This means the model is more likely to produce repetitive and predictable responses, closely adhering to the highest probability predictions. Lower temperatures are ideal for tasks requiring high precision and consistency, such as technical documentation or legal text generation, where accuracy and clarity are paramount.

- **Higher Temperature Values**: Conversely, when the temperature is increased (e.g., 1.0 or higher), the model's output becomes more random and creative. Higher temperatures introduce more variability by allowing the model to explore a wider range of potential outcomes, even those with lower probabilities. This can result in more diverse and imaginative responses, which is beneficial for creative writing, brainstorming, or generating unique content where novelty and originality are desired.

By tuning the temperature parameter, users can strike a balance between determinism and creativity, tailoring the model's behavior to suit specific use cases and desired outcomes.

#### max_tokens

- **Type**: `int`
- **Default**: `4096`
- **Example**: `131072`

**Description**:
Defines the maximum number of tokens that the model can generate in a single response. This limits the length of the generated output.

#### top_k

- **Type**: `int`
- **Example**: `40`

**Description**:
Enables top-k sampling, a technique used to control the randomness of the model's output. When generating each token, the model considers only the top k tokens from the probability distribution of possible next tokens. By focusing on the most probable tokens, this method reduces the likelihood of selecting unlikely or irrelevant words, thereby enhancing the coherence and relevance of the generated text.

This can be particularly useful in scenarios where maintaining a certain level of consistency and accuracy is crucial, such as in technical documentation, customer service responses, or any context where the precision of language is important. Adjusting the top_k parameter allows for fine-tuning the balance between creativity and reliability in the model's responses.

#### top_p

- **Type**: `float`
- **Example**: `0.9`

**Description**:
Enables nucleus sampling, also known as top-p sampling. This technique differs from top-k sampling by considering the smallest set of tokens whose cumulative probability meets or exceeds a specified threshold p. Instead of limiting the choice to a fixed number of top tokens, top-p sampling dynamically selects from a variable number of tokens based on their cumulative probability. This method allows for a more flexible and adaptive approach to token selection, balancing the likelihood of high-probability tokens with the inclusion of less common words.

By adjusting the top_p parameter, users can control the diversity and creativity of the model's output, making it possible to generate responses that are both coherent and varied, suitable for applications where a mix of predictability and innovation is desired.

#### typical_p

- **Type**: `float`
- **Example**: `0.8`

**Description**:
Typical sampling focuses on generating tokens that are most representative or average in their probability distribution. Unlike methods that solely prioritize the highest probability tokens, typical sampling strikes a balance between high-probability and diverse outputs. This approach ensures that the generated text is coherent and contextually appropriate while still allowing for a degree of variability and creativity. By adjusting the typical_p parameter, users can influence how closely the model's output adheres to expected patterns versus exploring less common but potentially more interesting choices.

This method is particularly useful in applications where maintaining a natural and fluent text generation is critical, while still incorporating some level of originality.

#### format

- **Type**: `str`
- **Example**: `"text-generation"`

**Description**:
Specifies the desired format of the output generated by the model. The format determines how the model processes and structures the generated content. For instance, setting the format to `"text-generation"` directs the model to produce coherent text sequences suitable for tasks such as writing paragraphs, generating dialogue, or completing sentences. This parameter can be tailored to fit different use cases, allowing users to optimize the model's output for specific applications such as text summarization, question answering, json or code generation.

By specifying the appropriate format, users can ensure that the generated content aligns with their requirements and use case scenarios. Take note that support for this parameter is decided by the speciffic model you are using, as must be added to the model during training.

#### repetition_penalty

- **Type**: `float`
- **Default**: `1.03`
- **Example**: `1.2`

**Description**:
A parameter that reduces the likelihood of repeating the same token or sequence of tokens. Higher values penalize repetition more strongly. This helps in generating more diverse and less repetitive outputs by discouraging the model from producing identical or very similar phrases.

Setting a higher repetition penalty can be particularly useful in tasks where variety and creativity are desired, such as story generation or creative writing, while lower values might be preferable for tasks requiring more precise and consistent information, such as factual summaries or technical documentation.

#### tfs_z

- **Type**: `float`
- **Default**: `1`
- **Example**: `0.5`

**Description**:
Tail-Free Sampling (TFS) parameter, used to control the diversity of the output. A value of 1 represents no additional sampling, producing outputs in their most predictable form. Values less than 1 introduce more diversity by allowing the model to sample less probable tokens, thereby increasing the variability and creativity of the generated text. Lower TFS values can be beneficial for creative writing or generating varied responses, while higher values (closer to 1) are suitable for more deterministic and consistent outputs.

These parameters provide users with granular control over the model's behavior, enabling customization for various applications and use cases. By adjusting these settings, users can fine-tune the balance between creativity, coherence, and specificity in the model's responses.

### Context Window

The context window in large language models (LLMs) refers to the maximum span of text the model can process at one time, measured in tokens. This size determines how much text the model can consider for generating responses, impacting its ability to maintain coherence and context over long passages. Larger context windows enable better handling of extended text sequences, making models more effective for tasks involving long documents and detailed conversations. Different models have varying context window sizes, influencing their performance on specific natural language processing tasks.

### Quantization

Quantization in the context of large language models (LLMs) refers to the process of reducing the precision of the model’s parameters to lower bit-width representations, such as converting 32-bit floating-point numbers to 8-bit integers. This technique significantly reduces the model size and computational requirements, allowing for faster inference and lower memory usage. 

Quantization can be particularly useful in deploying models on devices with limited resources, such as mobile phones or edge devices, where computational power and memory are constrained. While quantization can lead to some loss in model accuracy, it often provides a favorable trade-off between performance and efficiency.

There are various levels of quantization, such as:

- **Full-Precision (FP16/FP32):** High precision with larger model sizes and higher computational requirements.
- **8-bit Quantization (Q8):** Reduces model size and computational load, with minor accuracy loss.
- **6-bit Quantization (Q6):** Further reduces size and load, potentially with more significant accuracy trade-offs.
- **4-bit Quantization (Q4):** Even further reduces size and load, potentially with even more significant accuracy trade-offs.
- **2-bit Quantization (Q2):** Maximizes efficiency, but may result in more substantial accuracy loss.

By selecting the appropriate level of quantization, you can optimize LLM performance to match your specific application needs and hardware capabilities.

### Model Reference

In the context of the ModelFactory, the **model reference** is a crucial string identifier that specifies the exact configuration of a model to be used. This reference string is constructed using a specific naming convention that captures key attributes of the model, including the provider, the number of parameters, the context window, and the quantization level.

#### Naming Convention for Model Reference

For open-source models, the model reference string typically follows this format:

**"[short_provider_name]-[parameters]-[context_window]-[quantization]"**

This convention ensures that the model's characteristics are immediately identifiable, aiding users in selecting and modifying models efficiently. For example, the model reference string for the Phi3 model with 3 billion parameters, a 128k context window, and 6-bit quantization from Ollama would be:

**"o-phi3-3b-128k-q6"**
This structure allows users to quickly understand the specific attributes of the model being used, making it easier to make informed decisions about model selection and adjustments.

#### Example: Adjustments

Switching to a different model configuration is straightforward. By modifying the relevant parts of the model reference string, users can change models with minimal effort. For instance, changing the previous example to a 14 billion parameter version while keeping the same context window and quantization would result in:

**"o-phi3-14b-128k-q6"**
This simplicity in model referencing enhances usability, enabling quick experimentation and optimization.

#### Closed-Source Models

For closed-source models provided by companies like OpenAI, Google, and Anthropic, the model reference string often aligns with the naming conventions used by the providers themselves. This approach minimizes confusion and aligns with familiar reference standards used by these companies.

#### Usage in Code

The model reference string is used when calling models within the ModelFactory. Here’s an example of how it is implemented in code:

```python
from model_factory import ModelFactory

factory = ModelFactory()
llm = factory.get_model(model="model_reference")
```

In this example, replace `"model_reference"` with the actual reference string of the model you intend to use. This method ensures that the specified model is loaded with the correct parameters, context window, and quantization settings.

By adhering to these conventions, the ModelFactory provides a robust and user-friendly framework for managing and utilizing various models, enabling users to optimize their model selection and usage effectively.

## Google Models

[Intro to the google generative ai models here]

### Google Models - Supported Parameters

param model: str [Required]
   Model name to use.

param temperature: float = 0.7
   Run inference with this temperature. Must by in the closed interval [0.0, 1.0].

param google_api_key: Optional[SecretStr] = None¶

param max_output_tokens: Optional[int] = None
   Maximum number of tokens to include in a candidate. Must be greater than zero. If unset, will default to 64.

param max_retries: int = 6
   The maximum number of retries to make when generating.

param safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None
   The default safety settings to use for all generations.

   For example:
      safety_settings = {
         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH, HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
      }

param timeout: Optional[float] = None
   The maximum number of seconds to wait for a response.

param top_k: Optional[int] = None
   Decode using top-k sampling: consider the set of top_k most probable tokens. Must be positive.

param top_p: Optional[float] = None
   Decode using nucleus sampling: consider the smallest set of tokens whose probability sum is at least top_p. Must be in the closed interval [0.0, 1.0].

## OpenAI Models

[Intro to the openai ai models here]

### OpenAI Models Supported Parameters

param model_name: str = 'gpt-3.5-turbo' (alias 'model')
   Model name to use.

param openai_api_base: Optional[str] = None (alias 'base_url')
   Base URL path for API requests, leave blank if not using a proxy or service emulator.

param openai_api_key: Optional[str] = None (alias 'api_key')
   Automatically inferred from env var OPENAI_API_KEY if not provided.

param openai_organization: Optional[str] = None (alias 'organization')
   Automatically inferred from env var OPENAI_ORG_ID if not provided.

param temperature: float = 0.7
   What sampling temperature to use.

param max_retries: int = 2
   Maximum number of retries to make when generating.

param max_tokens: Optional[int] = None
   Maximum number of tokens to generate.

## Anthropic Models

[Intro to the anthropic models here]

### Anthropic Models Supported Parameters

param model: str [Required]
   Name of Anthropic model to use. E.g. “claude-3-sonnet-20240229”.

param api_key: Optional[str]
   Anthropic API key. If not passed in will be read from env var ANTHROPIC_API_KEY.

param base_url: Optional[str]
   Base URL for API requests. Only specify if using a proxy or service emulator.

param temperature: Optional[float]
   Sampling temperature. Ranges from 0.0 to 1.0.

param max_tokens: Optional[int]
   Max number of tokens to generate.

param timeout: Optional[float]
   Timeout for requests.

param max_retries: Optional[int]
   Max number of retries if a request fails.

## Groq Models

[Intro to the groq models here]

### Groq Models Supported Parameters

param model: str [Required]
   Name of Groq model to use. E.g. “mixtral-8x7b-32768”.

param api_key: Optional[str]
   Groq API key. If not passed in will be read from env var GROQ_API_KEY.

param base_url: Optional[str]
   Base URL path for API requests, leave blank if not using a proxy or service emulator.

param temperature: Optional[float]
   Sampling temperature. Ranges from 0.0 to 1.0.

param max_tokens: Optional[int]
   Max number of tokens to generate.

param max_retries: int
   Max number of retries.

## Ollama Models

Ollama has made a lot of models available for you to download and run locally.

### Ollama Models Supported Parameters

param model: str = 'llama2'
   Model name to use.

param base_url: str = 'http://localhost:11434'
   Base url the model is hosted under.

param temperature: Optional[float] = None
   The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)

param format: Optional[str] = None
   Specify the format of the output (e.g., json)

param num_ctx: Optional[int] = None
   Sets the size of the context window used to generate the next token. (Default: 4096)

param num_predict: Optional[int] = None
   Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)

param repeat_last_n: Optional[int] = None
Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)

param repeat_penalty: Optional[float] = None
   Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)

param top_k: Optional[int] = None
   Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)

param top_p: Optional[float] = None
   Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)

param tfs_z: Optional[float] = None
   Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)

param raw: Optional[bool] = None
   raw or not.”” The parameter (Default: 5 minutes) can be set to: 1. a duration string in Golang (such as “10m” or “24h”); 2. a number in seconds (such as 3600); 3. any negative number which will keep the model loaded in memory (e.g. -1 or “-1m”); 4. 0 which will unload the model immediately after generating a response; See the [Ollama documents](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately)

param system: Optional[str] = None
   system prompt (overrides what is defined in the Modelfile)

param template: Optional[str] = None
   full prompt or prompt template (overrides what is defined in the Modelfile)

### Phi3

**Multimodal Capabilities**: No

#### Phi3 Description

Phi3 is a highly capable model, trained on 14 billion parameters. It offers different context window sizes and quantization options to suit various needs. The model is designed to handle a wide range of natural language processing tasks, providing robust performance and flexibility. For detailed information, you can visit the [Phi3 model card](https://ollama.com/library/phi3).

#### Phi3 Model Variations

| Model Reference          | Parameters | Context Window | Quantization | Download Size | Model Card                                                                          |
|:-------------------------|-----------:|--------------:|--------------:|--------------:|:------------------------------------------------------------------------------------|
| o-phi3-14b-128k-fp16     |        14b |         128k  |          fp16 |          28GB | [Phi3 14b 128k fp16](https://ollama.com/library/phi3:14b-medium-128k-instruct-f16)  |
| o-phi3-14b-128k-q6       |        14b |         128k  |            q6 |          11GB | [Phi3 14b 128k q6](https://ollama.com/library/phi3:14b-medium-128k-instruct-q6_K)   |
| o-phi3-14b-128k-q4       |        14b |         128k  |            q4 |         8.6GB | [Phi3 14b 128k q4](https://ollama.com/library/phi3:14b-medium-128k-instruct-q4_K_M) |
| o-phi3-14b-128k-q2       |        14b |         128k  |            q2 |         5.1GB | [Phi3 14b 128k q2](https://ollama.com/library/phi3:14b-medium-128k-instruct-q2_K)   |
| o-phi3-14b-4k-fp16       |        14b |           4k  |          fp16 |          28GB | [Phi3 14b 4k fp16](https://ollama.com/library/phi3:14b-medium-4k-instruct-f16)      |
| o-phi3-14b-4k-q8         |        14b |           4k  |            q8 |          15GB | [Phi3 14b 4k q8](https://ollama.com/library/phi3:14b-medium-4k-instruct-q8_0)       |
| o-phi3-14b-4k-q6         |        14b |           4k  |            q6 |          11GB | [Phi3 14b 4k q6](https://ollama.com/library/phi3:14b-medium-4k-instruct-q6_K)       |
| o-phi3-14b-4k-q4         |        14b |           4k  |            q4 |         8.6GB | [Phi3 14b 4k q4](https://ollama.com/library/phi3:14b-medium-4k-instruct-q4_K_M)     |
| o-phi3-14b-4k-q2         |        14b |           4k  |            q2 |         5.1GB | [Phi3 14b 4k q2](https://ollama.com/library/phi3:14b-medium-4k-instruct-q2_K)       |
| o-phi3-3b-128k-fp16      |         3b |         128k  |          fp16 |         7.6GB | [Phi3 3b 128k fp16](https://ollama.com/library/phi3:3.8b-mini-128k-instruct-f16)    |
| o-phi3-3b-128k-q8        |         3b |         128k  |            q8 |         4.1GB | [Phi3 3b 128k q8](https://ollama.com/library/phi3:3.8b-mini-128k-instruct-q8_0)     |
| o-phi3-3b-128k-q4        |         3b |         128k  |            q4 |         2.4GB | [Phi3 3b 128k q4](https://ollama.com/library/phi3:3.8b-mini-128k-instruct-q4_1)     |
| o-phi3-3b-128k-q2        |         3b |         128k  |            q2 |         1.4GB | [Phi3 3b 128k q2](https://ollama.com/library/phi3:3.8b-mini-128k-instruct-q2_K)     |
| o-phi3-3b-4k-fp16        |         3b |           4k  |          fp16 |         7.6GB | [Phi3 3b 4k fp16](https://ollama.com/library/phi3:3.8b-mini-4k-instruct-f16)        |
| o-phi3-3b-4k-q8          |         3b |           4k  |            q8 |         4.1GB | [Phi3 3b 4k q8](https://ollama.com/library/phi3:3.8b-mini-4k-instruct-q8_0)         |
| o-phi3-3b-4k-q4          |         3b |           4k  |            q4 |         2.4GB | [Phi3 3b 4k q4](https://ollama.com/library/phi3:3.8b-mini-4k-instruct-q4_1)         |
| o-phi3-3b-4k-q2          |         3b |           4k  |            q2 |         1.4GB | [Phi3 3b 4k q2](https://ollama.com/library/phi3:3.8b-mini-4k-instruct-q2_K)         |


#### Phi3 Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-phi3-14b-128k-fp16")
```

#### Phi3 Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### Phi3 System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. You should test the models on your own systems to determine compatibility.

   Memory: Estimated 16GB RAM for high-performance models. For lower quantized models (e.g., q2, q4), lower memory configurations might suffice.
   Storage: Approximately 30GB free disk space for the full model. Lower quantized versions will require less storage.
   Processor: Modern multi-core CPU (e.g., Intel i7 or AMD Ryzen 7) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA RTX 3060 or higher).

#### Phi3 Additional Information

Phi3 models are versatile and can be used for a variety of natural language processing tasks. They are particularly well-suited for applications that require high accuracy and robustness, such as content generation, summarization, and language translation. The different quantization options allow for flexibility in balancing performance and resource usage.

### Qwen2

**Multimodal Capabilities**: No

#### Qwen2 Description

Qwen2 is a high-capacity model provided by Ollama, trained on 72 billion parameters. It offers a substantial context window of 128k tokens and various quantization options to cater to different performance and resource requirements. This model is designed to handle a wide range of natural language processing tasks with high efficiency and accuracy. For detailed information, you can visit the [Qwen2 model card](https://ollama.com/library/qwen2).

#### Qwen2 Variations

| Model Reference         | Parameters | Context Window | Quantization | Download Size | Model Card                                                                            |
|:------------------------|-----------:|---------------:|-------------:|--------------:|:--------------------------------------------------------------------------------------|
| o-qwn2-72b-128k-fp16    |        72b |           128k |         fp16 |         145GB | [Qwen2 72b 128k fp16](https://ollama.com/library/qwen2:72b-instruct-fp16)             |
| o-qwn2-72b-128k-q8      |        72b |           128k |           q8 |          77GB | [Qwen2 72b 128k q8](https://ollama.com/library/qwen2:72b-instruct-q8_0)               |
| o-qwn2-72b-128k-q6      |        72b |           128k |           q6 |          64GB | [Qwen2 72b 128k q6](https://ollama.com/library/qwen2:72b-instruct-q6_K)               |
| o-qwn2-72b-128k-q4      |        72b |           128k |           q4 |          47GB | [Qwen2 72b 128k q4](https://ollama.com/library/qwen2:72b-instruct-q4_K_M)             |
| o-qwn2-72b-128k-q2      |        72b |           128k |           q2 |          30GB | [Qwen2 72b 128k q2](https://ollama.com/library/qwen2:72b-instruct-q2_K)               |
| o-qwn2-7b-128k-fp16     |         7b |           128k |         fp16 |          15GB | [Qwen2 7b 128k fp16](https://ollama.com/library/qwen2:7b-instruct-fp16)               |
| o-qwn2-7b-128k-q8       |         7b |           128k |           q8 |         8.1GB | [Qwen2 7b 128k q8](https://ollama.com/library/qwen2:7b-instruct-q8_0)                 |
| o-qwn2-7b-128k-q6       |         7b |           128k |           q6 |         6.3GB | [Qwen2 7b 128k q6](https://ollama.com/library/qwen2:7b-instruct-q6_K)                 |
| o-qwn2-7b-128k-q4       |         7b |           128k |           q4 |         4.4GB | [Qwen2 7b 128k q4](https://ollama.com/library/qwen2:7b-instruct-q4_0)                 |
| o-qwn2-7b-128k-q2       |         7b |           128k |           q2 |           3GB | [Qwen2 7b 128k q2](https://ollama.com/library/qwen2:7b-instruct-q2_K)                 |
| o-qwn2-2b-32k-fp16      |         2b |            32k |         fp16 |         3.1GB | [Qwen2 2b 32k fp16](https://ollama.com/library/qwen2:1.5b-instruct-fp16)              |
| o-qwn2-2b-32k-q8        |         2b |            32k |           q8 |         1.6GB | [Qwen2 2b 32k q8](https://ollama.com/library/qwen2:1.5b-instruct-q8_0)                |
| o-qwn2-2b-32k-q6        |         2b |            32k |           q6 |         1.3GB | [Qwen2 2b 32k q6](https://ollama.com/library/qwen2:1.5b-instruct-q6_K)                |
| o-qwn2-2b-32k-q4        |         2b |            32k |           q4 |         935MB | [Qwen2 2b 32k q4](https://ollama.com/library/qwen2:1.5b-instruct-q4_0)                |
| o-qwn2-2b-32k-q2        |         2b |            32k |           q2 |         676MB | [Qwen2 2b 32k q2](https://ollama.com/library/qwen2:1.5b-instruct-q2_K)                |
| o-qwn2-1b-32k-fp16      |         1b |            32k |         fp16 |         994MB | [Qwen2 1b 32k fp16](https://ollama.com/library/qwen2:0.5b-instruct-fp16)              |
| o-qwn2-1b-32k-q8        |         1b |            32k |           q8 |         531MB | [Qwen2 1b 32k q8](https://ollama.com/library/qwen2:0.5b-instruct-q8_0)                |
| o-qwn2-1b-32k-q6        |         1b |            32k |           q6 |         506MB | [Qwen2 1b 32k q6](https://ollama.com/library/qwen2:0.5b-instruct-q6_K)                |
| o-qwn2-1b-32k-q4        |         1b |            32k |           q4 |         352MB | [Qwen2 1b 32k q4](https://ollama.com/library/qwen2:0.5b-instruct-q4_0)                |
| o-qwn2-1b-32k-q2        |         1b |            32k |           q2 |         339MB | [Qwen2 1b 32k q2](https://ollama.com/library/qwen2:0.5b-instruct-q2_K)                |

#### Qwen2 Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-qwn2-72b-128k-fp16")
```

#### Qwen2 Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### Qwen2 System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. Users should test the models on their own systems to determine compatibility.

- **Memory:** Estimated 128GB RAM for high-performance models. For lower quantized models (e.g., q2, q4), lower memory configurations might suffice.
- **Storage:** Up to 150GB free disk space for the full model. Lower quantized versions will require less storage.
- **Processor:** Modern multi-core CPU (e.g., Intel i9 or AMD Ryzen 9) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA RTX 3090 or higher).

#### Qwen2 Additional Information

Qwen2 models are versatile and can be used for a variety of natural language processing tasks. They are particularly well-suited for applications that require high accuracy and robustness, such as content generation, summarization, and language translation. The different quantization options allow for flexibility in balancing performance and resource usage.

### Llama 3

**Multimodal Capabilities**: No

#### Llama 3 Description

Llama 3 is a high-capacity model provided by Ollama, trained on 70 billion parameters. It offers a substantial context window of 8k tokens and various quantization options to cater to different performance and resource requirements. This model is designed to handle a wide range of natural language processing tasks with high efficiency and accuracy. For detailed information, you can visit the [Llama 3 model card](https://ollama.com/library/llama3).

#### Llama 3 Variations

| Model Reference         | Parameters | Context Window | Quantization | Download Size | Model Card                                                                             |
|:------------------------|-----------:|---------------:|-------------:|--------------:|:---------------------------------------------------------------------------------------|
| o-lma3-70b-8k-fp16      |        70b |             8k |         fp16 |         141GB | [Llama 3 70b 8k fp16](https://ollama.com/library/llama3:70b-instruct-fp16)             |
| o-lma3-70b-8k-q8        |        70b |             8k |           q8 |          75GB | [Llama 3 70b 8k q8](https://ollama.com/library/llama3:70b-instruct-q8_0)               |
| o-lma3-70b-8k-q6        |        70b |             8k |           q6 |          58GB | [Llama 3 70b 8k q6](https://ollama.com/library/llama3:70b-instruct-q6_K)               |
| o-lma3-70b-8k-q4        |        70b |             8k |           q4 |          40GB | [Llama 3 70b 8k q4](https://ollama.com/library/llama3:70b-instruct-q4_0)               |
| o-lma3-70b-8k-q2        |        70b |             8k |           q2 |          26GB | [Llama 3 70b 8k q2](https://ollama.com/library/llama3:70b-instruct-q2_K)               |
| o-lma3-8b-8k-fp16       |         8b |             8k |         fp16 |          16GB | [Llama 3 8b 8k fp16](https://ollama.com/library/llama3:8b-instruct-fp16)               |
| o-lma3-8b-8k-q8         |         8b |             8k |           q8 |         8.5GB | [Llama 3 8b 8k q8](https://ollama.com/library/llama3:8b-instruct-q8_0)                 |
| o-lma3-8b-8k-q6         |         8b |             8k |           q6 |         6.6GB | [Llama 3 8b 8k q6](https://ollama.com/library/llama3:8b-instruct-q6_K)                 |
| o-lma3-8b-8k-q4         |         8b |             8k |           q4 |         4.7GB | [Llama 3 8b 8k q4](https://ollama.com/library/llama3:8b-instruct-q4_0)                 |
| o-lma3-8b-8k-q2         |         8b |             8k |           q2 |         3.2GB | [Llama 3 8b 8k q2](https://ollama.com/library/llama3:8b-instruct-q2_K)                 |

#### Llama 3 Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-lma3-70b-8k-fp16")
```

#### Llama 3 Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### Llama 3 System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. Users should test the models on their own systems to determine compatibility.

- **Memory:** Estimated 32GB RAM for high-performance models. For lower quantized models (e.g., q2, q4), lower memory configurations might suffice.
- **Storage:** Approximately 150GB free disk space for the full model. Lower quantized versions will require less storage.
- **Processor:** Modern multi-core CPU (e.g., Intel i9 or AMD Ryzen 9) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA RTX 3090 or higher).

#### Llama 3 Additional Information

Llama 3 models are versatile and can be used for a variety of natural language processing tasks. They are particularly well-suited for applications that require high accuracy and robustness, such as content generation, summarization, and language translation. The different quantization options allow for flexibility in balancing performance and resource usage.

### Mistral

**Multimodal Capabilities**: No

#### Mistral Description

Mistral is a robust model provided by Ollama, trained on 7 billion parameters. It offers a context window of 32k tokens and various quantization options to accommodate different performance and resource requirements. This model is designed to handle a wide range of natural language processing tasks with high efficiency and accuracy. For detailed information, you can visit the [Mistral model card](https://ollama.com/library/mistral).

#### Mistral Variations

| Model Reference         | Parameters | Context Window | Quantization | Download Size | Model Card                                                                                  |
|:------------------------|-----------:|---------------:|-------------:|--------------:|:--------------------------------------------------------------------------------------------|
| o-mis-7b-32k-fp16       |         7b |            32k |         fp16 |          14GB | [Mistral 7b 32k fp16](https://ollama.com/library/mistral:7b-instruct-v0.3-fp16)             |
| o-mis-7b-32k-q8         |         7b |            32k |           q8 |         7.7GB | [Mistral 7b 32k q8](https://ollama.com/library/mistral:7b-instruct-v0.3-q8_0)               |
| o-mis-7b-32k-q6         |         7b |            32k |           q6 |         5.9GB | [Mistral 7b 32k q6](https://ollama.com/library/mistral:7b-instruct-v0.3-q6_K)               |
| o-mis-7b-32k-q4         |         7b |            32k |           q4 |         4.1GB | [Mistral 7b 32k q4](https://ollama.com/library/mistral:7b-instruct-v0.3-q4_0)               |
| o-mis-7b-32k-q2         |         7b |            32k |           q2 |         2.7GB | [Mistral 7b 32k q2](https://ollama.com/library/mistral:7b-instruct-v0.3-q2_K)               |

#### Mistral Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-mis-7b-32k-fp16")
```

#### Mistral Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### Mistral System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. Users should test the models on their own systems to determine compatibility.

- **Memory:** Estimated 16GB RAM for high-performance models. For lower quantized models (e.g., q2, q4), lower memory configurations might suffice.
- **Storage:** Approximately 20GB free disk space for the full model. Lower quantized versions will require less storage.
- **Processor:** Modern multi-core CPU (e.g., Intel i7 or AMD Ryzen 7) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA RTX 3080 or higher).

#### Mistral Additional Information

Mistral models are versatile and can be used for a variety of natural language processing tasks. They are particularly well-suited for applications that require high accuracy and robustness, such as content generation, summarization, and language translation. The different quantization options allow for flexibility in balancing performance and resource usage.

### Mixtral

**Multimodal Capabilities**: No

#### Mixtral Description

Mixtral is an advanced model provided by Ollama, trained on 8 instances of 22 billion parameters each. It offers a substantial context window of 64k tokens and various quantization options to cater to different performance and resource requirements. This model is designed to handle a wide range of natural language processing tasks with high efficiency and accuracy. For detailed information, you can visit the [Mixtral model card](https://ollama.com/library/mixtral).

#### Mixtral Variations

| Model Reference         | Parameters | Context Window | Quantization | Download Size | Model Card                                                                                       |
|:------------------------|-----------:|---------------:|-------------:|--------------:|:-------------------------------------------------------------------------------------------------|
| o-mix-8x22b-64k-fp16    |      8x22b |            64k |         fp16 |         281GB | [Mixtral 8x22b 64k fp16](https://ollama.com/library/mixtral:8x22b-instruct-v0.1-fp16)            |
| o-mix-8x22b-64k-q8      |      8x22b |            64k |           q8 |         149GB | [Mixtral 8x22b 64k q8](https://ollama.com/library/mixtral:8x22b-instruct-v0.1-q8_0)              |
| o-mix-8x22b-64k-q6      |      8x22b |            64k |           q6 |         116GB | [Mixtral 8x22b 64k q6](https://ollama.com/library/mixtral:8x22b-instruct-v0.1-q6_K)              |
| o-mix-8x22b-64k-q4      |      8x22b |            64k |           q4 |          80GB | [Mixtral 8x22b 64k q4](https://ollama.com/library/mixtral:8x22b-instruct-v0.1-q4_0)              |
| o-mix-8x22b-64k-q2      |      8x22b |            64k |           q2 |          52GB | [Mixtral 8x22b 64k q2](https://ollama.com/library/mixtral:8x22b-instruct-v0.1-q2_K)              |
| o-mix-8x7b-32k-fp16     |       8x7b |            32k |         fp16 |          93GB | [Mixtral 8x7b 32k fp16](https://ollama.com/library/mixtral:8x7b-instruct-v0.1-fp16)              |
| o-mix-8x7b-32k-q8       |       8x7b |            32k |           q8 |          50GB | [Mixtral 8x7b 32k q8](https://ollama.com/library/mixtral:8x7b-instruct-v0.1-q8_0)                |
| o-mix-8x7b-32k-q6       |       8x7b |            32k |           q6 |          38GB | [Mixtral 8x7b 32k q6](https://ollama.com/library/mixtral:8x7b-instruct-v0.1-q6_K)                |
| o-mix-8x7b-32k-q4       |       8x7b |            32k |           q4 |          26GB | [Mixtral 8x7b 32k q4](https://ollama.com/library/mixtral:8x7b-instruct-v0.1-q4_0)                |
| o-mix-8x7b-32k-q2       |       8x7b |            32k |           q2 |          16GB | [Mixtral 8x7b 32k q2](https://ollama.com/library/mixtral:8x7b-instruct-v0.1-q2_K)                |

#### Mixtral Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-mix-8x22b-64k-fp16")
```

#### Mixtral Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### Mixtral System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. Users should test the models on their own systems to determine compatibility.

- **Memory:** Estimated 64GB RAM for high-performance models. For lower quantized models (e.g., q2, q4), lower memory configurations might suffice.
- **Storage:** Approximately 300GB free disk space for the full model. Lower quantized versions will require less storage.
- **Processor:** Modern multi-core CPU (e.g., Intel i9 or AMD Ryzen 9) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA RTX 3090 or higher).

#### Mixtral Additional Information

Mixtral models are versatile and can be used for a variety of natural language processing tasks. They are particularly well-suited for applications that require high accuracy and robustness, such as content generation, summarization, and language translation. The different quantization options allow for flexibility in balancing performance and resource usage.

### Gemma

**Multimodal Capabilities**: No

#### Gemma Description

Gemma is a robust model provided by Ollama, trained on 7 billion parameters. It offers a context window of 8k tokens and various quantization options to accommodate different performance and resource requirements. This model is designed to handle a wide range of natural language processing tasks with high efficiency and accuracy. For detailed information, you can visit the [Gemma model card](https://ollama.com/library/gemma).

#### Gemma Variations

| Model Reference         | Parameters | Context Window | Quantization | Download Size | Model Card                                                      |
|-------------------------|-----------:|---------------:|-------------:|--------------:|-----------------------------------------------------------------|
| o-gma-7b-8k-fp16        |         7b |             8k |         fp16 |          17GB | [Gemma 7b 8k fp16](https://ollama.com/library/gemma:7b-instruct-v1.1-fp16)             |
| o-gma-7b-8k-q8          |         7b |             8k |           q8 |         9.1GB | [Gemma 7b 8k q8](https://ollama.com/library/gemma:7b-instruct-v1.1-q8_0)               |
| o-gma-7b-8k-q6          |         7b |             8k |           q6 |           7GB | [Gemma 7b 8k q6](https://ollama.com/library/gemma:7b-instruct-v1.1-q6_K)               |
| o-gma-7b-8k-q4          |         7b |             8k |           q4 |           5GB | [Gemma 7b 8k q4](https://ollama.com/library/gemma:7b-instruct-v1.1-q4_0)               |
| o-gma-7b-8k-q2          |         7b |             8k |           q2 |         3.5GB | [Gemma 7b 8k q2](https://ollama.com/library/gemma:7b-instruct-v1.1-q2_K)               |
| o-gma-2b-8k-fp16        |         2b |             8k |         fp16 |           5GB | [Gemma 2b 8k fp16](https://ollama.com/library/gemma:2b-instruct-v1.1-fp16)             |
| o-gma-2b-8k-q8          |         2b |             8k |           q8 |         2.7GB | [Gemma 2b 8k q8](https://ollama.com/library/gemma:2b-instruct-v1.1-q8_0)               |
| o-gma-2b-8k-q6          |         2b |             8k |           q6 |         2.1GB | [Gemma 2b 8k q6](https://ollama.com/library/gemma:2b-instruct-v1.1-q6_K)               |
| o-gma-2b-8k-q4          |         2b |             8k |           q4 |         1.6GB | [Gemma 2b 8k q4](https://ollama.com/library/gemma:2b-instruct-v1.1-q4_0)               |
| o-gma-2b-8k-q2          |         2b |             8k |           q2 |         1.2GB | [Gemma 2b 8k q2](https://ollama.com/library/gemma:2b-instruct-v1.1-q2_K)               |

#### Gemma Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-gma-7b-8k-fp16")
```

#### Gemma Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### Gemma System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. Users should test the models on their own systems to determine compatibility.

- **Memory:** Estimated 16GB RAM for high-performance models. For lower quantized models (e.g., q2, q4), lower memory configurations might suffice.
- **Storage:** Approximately 20GB free disk space for the full model. Lower quantized versions will require less storage.
- **Processor:** Modern multi-core CPU (e.g., Intel i7 or AMD Ryzen 7) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA RTX 3080 or higher).

#### Gemma Additional Information

Gemma models are versatile and can be used for a variety of natural language processing tasks. They are particularly well-suited for applications that require high accuracy and robustness, such as content generation, summarization, and language translation. The different quantization options allow for flexibility in balancing performance and resource usage.

### WizardLM-2

**Multimodal Capabilities**: No

#### WizardLM-2 Description

WizardLM-2 is a powerful model provided by Ollama, trained on 8 instances of 22 billion parameters each. It offers a substantial context window of 64k tokens and various quantization options to cater to different performance and resource requirements. This model is designed to handle a wide range of natural language processing tasks with high efficiency and accuracy. For detailed information, you can visit the [WizardLM-2 model card](https://ollama.com/library/wizardlm2).

#### WizardLM-2 Variations

| Model Reference         | Parameters | Context Window | Quantization | Download Size | Model Card                                                      |
|-------------------------|-----------:|---------------:|-------------:|--------------:|-----------------------------------------------------------------|
| o-wizlm2-8x22b-64k-fp16 |      8x22b |            64k |         fp16 |         281GB | [WizardLM-2 8x22b 64k fp16](https://ollama.com/library/wizardlm2:8x22b-fp16)            |
| o-wizlm2-8x22b-64k-q8   |      8x22b |            64k |           q8 |         149GB | [WizardLM-2 8x22b 64k q8](https://ollama.com/library/wizardlm2:8x22b-q8_0)              |
| o-wizlm2-8x22b-64k-q4   |      8x22b |            64k |           q4 |          80GB | [WizardLM-2 8x22b 64k q4](https://ollama.com/library/wizardlm2:8x22b-q4_0)              |
| o-wizlm2-8x22b-64k-q2   |      8x22b |            64k |           q2 |          52GB | [WizardLM-2 8x22b 64k q2](https://ollama.com/library/wizardlm2:8x22b-q2_K)              |
| o-wizlm2-7b-32k-fp16    |         7b |            32k |         fp16 |          14GB | [WizardLM-2 7b 32k fp16](https://ollama.com/library/wizardlm2:7b-fp16)                  |
| o-wizlm2-7b-32k-q8      |         7b |            32k |           q8 |         7.7GB | [WizardLM-2 7b 32k q8](https://ollama.com/library/wizardlm2:7b-q8_0)                    |
| o-wizlm2-7b-32k-q6      |         7b |            32k |           q6 |         5.9GB | [WizardLM-2 7b 32k q6](https://ollama.com/library/wizardlm2:7b-q6_K)                    |
| o-wizlm2-7b-32k-q4      |         7b |            32k |           q4 |         4.1GB | [WizardLM-2 7b 32k q4](https://ollama.com/library/wizardlm2:7b-q4_0)                    |
| o-wizlm2-7b-32k-q2      |         7b |            32k |           q2 |         2.7GB | [WizardLM-2 7b 32k q2](https://ollama.com/library/wizardlm2:7b-q2_K)                    |

#### WizardLM-2 Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-wizlm2-8x22b-64k-fp16")
```

#### WizardLM-2 Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### WizardLM-2 System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. Users should test the models on their own systems to determine compatibility.

- **Memory:** Estimated 64GB RAM for high-performance models. For lower quantized models (e.g., q2, q4), lower memory configurations might suffice.
- **Storage:** Approximately 300GB free disk space for the full model. Lower quantized versions will require less storage.
- **Processor:** Modern multi-core CPU (e.g., Intel i9 or AMD Ryzen 9) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA RTX 3090 or higher).

#### WizardLM-2 Additional Information

WizardLM-2 models are versatile and can be used for a variety of natural language processing tasks. They are particularly well-suited for applications that require high accuracy and robustness, such as content generation, summarization, and language translation. The different quantization options allow for flexibility in balancing performance and resource usage.

### Codestral

**Multimodal Capabilities**: No

#### Codestral Description

Codestral is a sophisticated model provided by Ollama, trained on 22 billion parameters. It offers a context window of 32k tokens and various quantization options to accommodate different performance and resource requirements. This model is designed to handle a wide range of natural language processing tasks with high efficiency and accuracy. For detailed information, you can visit the [Codestral model card](https://ollama.com/library/codestral).

#### Codestral Variations

| Model Reference         | Parameters | Context Window | Quantization | Download Size | Model Card                                                      |
|-------------------------|-----------:|---------------:|-------------:|--------------:|-----------------------------------------------------------------|
| o-cstral-22b-32k-fp16   |        22b |          32k   |         fp16 |          44GB | [Codestral 22b 32k fp16](https://ollama.com/library/codestral:22b-v0.1-f16)             |
| o-cstral-22b-32k-q8     |        22b |          32k   |           q8 |          24GB | [Codestral 22b 32k q8](https://ollama.com/library/codestral:22b-v0.1-q8_0)              |
| o-cstral-22b-32k-q6     |        22b |          32k   |           q6 |          18GB | [Codestral 22b 32k q6](https://ollama.com/library/codestral:22b-v0.1-q6_K)              |
| o-cstral-22b-32k-q4     |        22b |          32k   |           q4 |          13GB | [Codestral 22b 32k q4](https://ollama.com/library/codestral:22b-v0.1-q4_0)              |
| o-cstral-22b-32k-q2     |        22b |          32k   |           q2 |         8.3GB | [Codestral 22b 32k q2](https://ollama.com/library/codestral:22b-v0.1-q2_K)              |

#### Codestral Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-cstral-22b-32k-fp16")
```

#### Codestral Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### Codestral System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. Users should test the models on their own systems to determine compatibility.

- **Memory:** Estimated 32GB RAM for high-performance models. For lower quantized models (e.g., q2, q4), lower memory configurations might suffice.
- **Storage:** Approximately 50GB free disk space for the full model. Lower quantized versions will require less storage.
- **Processor:** Modern multi-core CPU (e.g., Intel i7 or AMD Ryzen 7) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA RTX 3080 or higher).

#### Codestral Additional Information

Codestral models are versatile and can be used for a variety of natural language processing tasks. They are particularly well-suited for applications that require high accuracy and robustness, such as content generation, summarization, and language translation. The different quantization options allow for flexibility in balancing performance and resource usage.

### LLaVA Llama 3

**Multimodal Capabilities**: Yes

#### LLaVA Llama 3 Description

LLaVA Llama 3 is a powerful and versatile model provided by Ollama, trained on 8 billion parameters. It offers a context window of 32k tokens and various quantization options to accommodate different performance and resource requirements. This model is designed to handle a wide range of natural language processing tasks with high efficiency and accuracy. For detailed information, you can visit the [LLaVA Llama 3 model card](https://ollama.com/library/llava-llama3).

#### LLaVA Llama 3 Variations

| Model Reference         | Parameters | Context Window | Quantization | Download Size | Model Card                                                      |
|-------------------------|-----------:|---------------:|-------------:|--------------:|-----------------------------------------------------------------|
| o-lva-8b-32k-fp16       |         8b |          32k   |         fp16 |          16GB | [LLaVA Llama 3 8b 32k fp16](https://ollama.com/library/llava-llama3:8b-v1.1-fp16)             |
| o-lva-8b-32k-q4         |         8b |          32k   |           q4 |         4.9GB | [LLaVA Llama 3 8b 32k q4](https://ollama.com/library/llava-llama3:8b-v1.1-q4_0)               |

#### LLaVA Llama 3 Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-lva-8b-32k-fp16")
```

#### LLaVA Llama 3 Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### LLaVA Llama 3 System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. Users should test the models on their own systems to determine compatibility.

- **Memory:** Estimated 32GB RAM for high-performance models. For lower quantized models (e.g., q4), lower memory configurations might suffice.
- **Storage:** Approximately 20GB free disk space for the full model. Lower quantized versions will require less storage.
- **Processor:** Modern multi-core CPU (e.g., Intel i7 or AMD Ryzen 7) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA RTX 3080 or higher).

#### LLaVA Llama 3 Additional Information

LLaVA Llama 3 models are versatile and can be used for a variety of natural language processing tasks. They are particularly well-suited for applications that require high accuracy and robustness, such as content generation, summarization, and language translation. The different quantization options allow for flexibility in balancing performance and resource usage.

### Gradient Llama 3

**Multimodal Capabilities**: No

#### Gradient Llama 3 Description

Gradient Llama 3 is a cutting-edge model provided by Ollama, trained on 70 billion parameters. It offers an extensive context window of 1 million tokens and various quantization options to accommodate different performance and resource requirements. This model is designed to handle a wide range of natural language processing tasks with high efficiency and accuracy. For detailed information, you can visit the [Gradient Llama 3 model card](https://ollama.com/library/llama3-gradient).

#### Gradient Llama 3 Variations

| Model Reference         | Parameters | Context Window | Quantization | Download Size | Model Card                                                      |
|-------------------------|-----------:|---------------:|-------------:|--------------:|-----------------------------------------------------------------|
| o-l3g-70b-1m-fp16       |        70b |            1m  |         fp16 |         141GB | [Gradient Llama 3 70b 1m fp16](https://ollama.com/library/llama3-gradient:70b-instruct-1048k-fp16)             |
| o-l3g-70b-1m-q8         |        70b |            1m  |           q8 |          75GB | [Gradient Llama 3 70b 1m q8](https://ollama.com/library/llama3-gradient:70b-instruct-1048k-q8_0)               |
| o-l3g-70b-1m-q6         |        70b |            1m  |           q6 |          58GB | [Gradient Llama 3 70b 1m q6](https://ollama.com/library/llama3-gradient:70b-instruct-1048k-q6_K)               |
| o-l3g-70b-1m-q4         |        70b |            1m  |           q4 |          40GB | [Gradient Llama 3 70b 1m q4](https://ollama.com/library/llama3-gradient:70b-instruct-1048k-q4_0)               |
| o-l3g-70b-1m-q2         |        70b |            1m  |           q2 |          26GB | [Gradient Llama 3 70b 1m q2](https://ollama.com/library/llama3-gradient:70b-instruct-1048k-q2_K)               |

#### Gradient Llama 3 Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-l3g-70b-1m-fp16")
```

#### Gradient Llama 3 Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### Gradient Llama 3 System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. Users should test the models on their own systems to determine compatibility.

- **Memory:** Estimated 64GB RAM for high-performance models. For lower quantized models (e.g., q2, q4), lower memory configurations might suffice.
- **Storage:** Approximately 150GB free disk space for the full model. Lower quantized versions will require less storage.
- **Processor:** Modern multi-core CPU (e.g., Intel i9 or AMD Ryzen 9) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA RTX 3090 or higher).

#### Gradient Llama 3 Additional Information

Gradient Llama 3 models are versatile and can be used for a variety of natural language processing tasks. They are particularly well-suited for applications that require high accuracy and robustness, such as content generation, summarization, and language translation. The different quantization options allow for flexibility in balancing performance and resource usage.

### LLaVA Phi3

**Multimodal Capabilities**: Yes

#### LLaVA Phi3 Description

LLaVA Phi3 is a highly efficient and versatile model provided by Ollama, trained on 3 billion parameters. It offers a context window of 4k tokens and various quantization options to accommodate different performance and resource requirements. This model is designed to handle a wide range of natural language processing tasks with high efficiency and accuracy. For detailed information, you can visit the [LLaVA Phi3 model card](https://ollama.com/library/llava-phi3).

#### LLaVA Phi3 Variations

| Model Reference         | Parameters | Context Window | Quantization | Download Size | Model Card                                                      |
|-------------------------|-----------:|---------------:|-------------:|--------------:|-----------------------------------------------------------------|
| o-lvaphi3-3b-4k-fp16    |         3b |            4k  |         fp16 |          7.6GB | [LLaVA Phi3 3b 4k fp16](https://ollama.com/library/llava-phi3:3.8b-mini-fp16)             |
| o-lvaphi3-3b-4k-q4      |         3b |            4k  |           q4 |          2.3GB | [LLaVA Phi3 3b 4k q4](https://ollama.com/library/llava-phi3:3.8b-mini-q4_0)               |

#### LLaVA Phi3 Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-lvaphi3-3b-4k-fp16")
```

#### LLaVA Phi3 Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### LLaVA Phi3 System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. Users should test the models on their own systems to determine compatibility.

- **Memory:** Estimated 16GB RAM for high-performance models. For lower quantized models (e.g., q4), lower memory configurations might suffice.
- **Storage:** Approximately 10GB free disk space for the full model. Lower quantized versions will require less storage.
- **Processor:** Modern multi-core CPU (e.g., Intel i7 or AMD Ryzen 7) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA RTX 3060 or higher).

#### LLaVA Phi3 Additional Information

LLaVA Phi3 models are versatile and can be used for a variety of natural language processing tasks. They are particularly well-suited for applications that require high accuracy and robustness, such as content generation, summarization, and language translation. The different quantization options allow for flexibility in balancing performance and resource usage.

### CodeGemma

**Multimodal Capabilities**: No

#### CodeGemma Description

CodeGemma is a collection of powerful, lightweight models that can perform a variety of coding tasks like fill-in-the-middle code completion, code generation, natural language understanding, mathematical reasoning, and instruction following.

Links:

- [CodeGemma model card](https://ollama.com/library/codegemma).
- [HuggingFace](https://huggingface.co/google/codegemma-7b)
- [Google Report](https://storage.googleapis.com/deepmind-media/gemma/codegemma_report.pdf)

#### CodeGemma Variations

| Model Reference         | Parameters | Context Window | Quantization | Download Size | Model Card                                                      |
|-------------------------|-----------:|---------------:|-------------:|--------------:|-----------------------------------------------------------------|
| o-cgma-7b-8k-fp16       |         7b |            8k  |         fp16 |          17GB | [CodeGemma 7b 8k fp16](https://ollama.com/library/codegemma:7b-code-fp16)             |
| o-cgma-7b-8k-q8         |         7b |            8k  |           q8 |         9.1GB | [CodeGemma 7b 8k q8](https://ollama.com/library/codegemma:7b-code-q8_0)               |
| o-cgma-7b-8k-q6         |         7b |            8k  |           q6 |           7GB | [CodeGemma 7b 8k q6](https://ollama.com/library/codegemma:7b-code-q6_K)               |
| o-cgma-7b-8k-q4         |         7b |            8k  |           q4 |           5GB | [CodeGemma 7b 8k q4](https://ollama.com/library/codegemma:7b-code-q4_0)               |
| o-cgma-7b-8k-q2         |         7b |            8k  |           q2 |         3.5GB | [CodeGemma 7b 8k q2](https://ollama.com/library/codegemma:7b-code-q2_K)               |

#### CodeGemma Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-cgma-7b-8k-fp16")
```

#### CodeGemma Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### CodeGemma System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. Users should test the models on their own systems to determine compatibility.

- **Memory:** Estimated 16GB RAM for high-performance models. For lower quantized models (e.g., q2, q4), lower memory configurations might suffice.
- **Storage:** Approximately 20GB free disk space for the full model. Lower quantized versions will require less storage.
- **Processor:** Modern multi-core CPU (e.g., Intel i7 or AMD Ryzen 7) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA RTX 3060 or higher).

#### CodeGemma Additional Information

For *code compleation* and *generation from natural languge*, use the **code version** (o-cgma). 
For *generation from natual language*, *chat* and *instruction following*, use the **instruct version** (o-cgmai).

### Command R

**Funtion Callon**: Yes
**Multimodal Capabilities**: No

#### Command R Description

Command R is a generative model optimized for long context tasks such as retrieval-augmented generation (RAG) and using external APIs and tools. As a model built for companies to implement at scale, Command R boasts:

- Strong accuracy on RAG and Tool Use
- Low latency, and high throughput
- Longer 128k context
- Strong capabilities across 10 key languages

Links:

- [Command R model card](https://ollama.com/library/command-r).
- [Cohere.com](https://cohere.com/blog/command-r)
- [HuggingFace](https://huggingface.co/CohereForAI/c4ai-command-r-v01)

#### Command R Variations

| Model Reference         | Parameters | Context Window | Quantization | Download Size | Model Card                                                      |
|-------------------------|-----------:|---------------:|-------------:|--------------:|-----------------------------------------------------------------|
| o-cmd-r-35b-128k-fp16   |        35b |          128k  |         fp16 |          70GB | [Command R 35b 128k fp16](https://ollama.com/library/command-r:35b-v0.1-fp16)             |
| o-cmd-r-35b-128k-q8     |        35b |          128k  |           q8 |          37GB | [Command R 35b 128k q8](https://ollama.com/library/command-r:35b-v0.1-q8_0)               |
| o-cmd-r-35b-128k-q6     |        35b |          128k  |           q6 |          29GB | [Command R 35b 128k q6](https://ollama.com/library/command-r:35b-v0.1-q6_K)               |
| o-cmd-r-35b-128k-q4     |        35b |          128k  |           q4 |          20GB | [Command R 35b 128k q4](https://ollama.com/library/command-r:35b-v0.1-q4_0)               |
| o-cmd-r-35b-128k-q2     |        35b |          128k  |           q2 |          14GB | [Command R 35b 128k q2](https://ollama.com/library/command-r:35b-v0.1-q2_K)               |

#### Command R Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-cmd-r-35b-128k-fp16")
```

#### Command R Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### Command R System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. Users should test the models on their own systems to determine compatibility.

- **Memory:** Estimated 64GB RAM for high-performance models. For lower quantized models (e.g., q2, q4), lower memory configurations might suffice.
- **Storage:** Approximately 75GB free disk space for the full model. Lower quantized versions will require less storage.
- **Processor:** Modern multi-core CPU (e.g., Intel i9 or AMD Ryzen 9) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA RTX 3090 or higher).

#### Command R Additional Information

Command R models are versatile and can be used for a variety of natural language processing tasks. They are particularly well-suited for applications that require high accuracy and robustness, such as content generation, summarization, and language translation. The different quantization options allow for flexibility in balancing performance and resource usage.

### Command R+

**Function Calling**: Yes
**Multimodal Capabilities**: No

#### Command R+ Description

Command R+ is Cohere’s most powerful, scalable large language model (LLM) purpose-built to excel at real-world enterprise use cases. Command R+ balances high efficiency with strong accuracy, enabling businesses to move beyond proof-of-concept, and into production with AI:

- A 128k-token context window
- Advanced Retrieval Augmented Generation (RAG) with citation to reduce hallucinations
- Multilingual coverage in 10 key languages to support global business operations
- Tool Use to automate sophisticated business processes

Links:

- [Command R+ model card](https://ollama.com/library/command-r-plus)
- [Cohere.com](https://cohere.com/blog/command-r-plus-microsoft-azure)
- [HuggingFace](https://huggingface.co/CohereForAI/c4ai-command-r-plus)


#### Command R+ Variations

| Model Reference         | Parameters | Context Window | Quantization | Download Size | Model Card                                                      |
|-------------------------|-----------:|---------------:|-------------:|--------------:|-----------------------------------------------------------------|
| o-cmd-rp-104b-128k-fp16 |       104b |          128k  |         fp16 |         208GB | [Command R+ 104b 128k fp16](https://ollama.com/library/command-r-plus:104b-fp16)             |
| o-cmd-rp-104b-128k-q8   |       104b |          128k  |           q8 |         110GB | [Command R+ 104b 128k q8](https://ollama.com/library/command-r-plus:104b-q8_0)               |
| o-cmd-rp-104b-128k-q4   |       104b |          128k  |           q4 |          59GB | [Command R+ 104b 128k q4](https://ollama.com/library/command-r-plus:104b-q4_0)               |
| o-cmd-rp-104b-128k-q2   |       104b |          128k  |           q2 |          39GB | [Command R+ 104b 128k q2](https://ollama.com/library/command-r-plus:104b-q2_K)               |

#### Command R+ Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-cmd-rp-104b-128k-fp16")
```

#### Command R+ Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### Command R+ System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. Users should test the models on their own systems to determine compatibility.

- **Memory:** Estimated 128GB RAM for high-performance models. For lower quantized models (e.g., q2, q4), lower memory configurations might suffice.
- **Storage:** Approximately 250GB free disk space for the full model. Lower quantized versions will require less storage.
- **Processor:** Modern multi-core CPU (e.g., Intel i9 or AMD Ryzen 9) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA RTX 3090 or higher).

#### Command R+ Additional Information

Command R+ models are versatile and can be used for a variety of natural language processing tasks. They are particularly well-suited for applications that require high accuracy and robustness, such as content generation, summarization, and language translation. The different quantization options allow for flexibility in balancing performance and resource usage.

### LLaVA

**Function Calling**: N/A
**Multimodal Capabilities**: Yes

#### LLaVA Description

LLaVA (Large Language-and-Vision Assistant) is a sophisticated multimodal model. It integrates a vision encoder with Vicuna, achieving high-level visual and language understanding capabilities, similar to the multimodal GPT-4. The model supports up to 34 billion parameters, offering different configurations to meet varied performance and resource requirements.

Key features of LLaVA include enhanced visual reasoning, improved OCR capabilities, and higher image resolution support, allowing the model to process more detailed visuals. These capabilities make LLaVA particularly well-suited for applications requiring large context windows and multimodal understanding, such as detailed image analysis, complex document comprehension, and robust visual conversations.

For detailed information, you can visit the [LLaVA model card](https://ollama.com/library/llava).

#### LLaVA Variations

| Model Reference         | Parameters | Context Window | Quantization | Download Size | Model Card                                                      |
|-------------------------|-----------:|---------------:|-------------:|--------------:|-----------------------------------------------------------------|
| o-lva-34b-4k-fp16       |        34b |            4k  |         fp16 |          69GB | [LLaVA 34b 4k fp16](https://ollama.com/library/llava:34b-v1.6-fp16)                |
| o-lva-34b-4k-q8         |        34b |            4k  |           q8 |          37GB | [LLaVA 34b 4k q8](https://ollama.com/library/llava:34b-v1.6-q8_0)                  |
| o-lva-34b-4k-q6         |        34b |            4k  |           q6 |          29GB | [LLaVA 34b 4k q6](https://ollama.com/library/llava:34b-v1.6-q6_K)                  |
| o-lva-34b-4k-q4         |        34b |            4k  |           q4 |          20GB | [LLaVA 34b 4k q4](https://ollama.com/library/llava:34b-v1.6-q4_0)                  |
| o-lva-34b-4k-q2         |        34b |            4k  |           q2 |          14GB | [LLaVA 34b 4k q2](https://ollama.com/library/llava:34b-v1.6-q2_K)                  |
| o-lva-13b-4k-fp16       |        13b |            4k  |         fp16 |          27GB | [LLaVA 13b 4k fp16](https://ollama.com/library/llava:13b-v1.6-vicuna-fp16)         |
| o-lva-13b-4k-q8         |        13b |            4k  |           q8 |          14GB | [LLaVA 13b 4k q8](https://ollama.com/library/llava:13b-v1.6-vicuna-q8_0)           |
| o-lva-13b-4k-q6         |        13b |            4k  |           q6 |          11GB | [LLaVA 13b 4k q6](https://ollama.com/library/llava:13b-v1.6-vicuna-q6_K)           |
| o-lva-13b-4k-q4         |        13b |            4k  |           q4 |           8GB | [LLaVA 13b 4k q4](https://ollama.com/library/llava:13b-v1.6-vicuna-q4_0)           |
| o-lva-13b-4k-q2         |        13b |            4k  |           q2 |         5.5GB | [LLaVA 13b 4k q2](https://ollama.com/library/llava:13b-v1.6-vicuna-q2_K)           |
| o-lva-7b-4k-fp16        |         7b |            4k  |         fp16 |          14GB | [LLaVA 7b 4k fp16](https://ollama.com/library/llava:7b-v1.6-vicuna-fp16)           |
| o-lva-7b-4k-q8          |         7b |            4k  |           q8 |         7.8GB | [LLaVA 7b 4k q8](https://ollama.com/library/llava:7b-v1.6-vicuna-q8_0)             |
| o-lva-7b-4k-q6          |         7b |            4k  |           q6 |         5.5GB | [LLaVA 7b 4k q6](https://ollama.com/library/llava:7b-v1.6-vicuna-q6_K)             |
| o-lva-7b-4k-q4          |         7b |            4k  |           q4 |         4.5GB | [LLaVA 7b 4k q4](https://ollama.com/library/llava:7b-v1.6-vicuna-q4_0)             |
| o-lva-7b-4k-q2          |         7b |            4k  |           q2 |         3.2GB | [LLaVA 7b 4k q2](https://ollama.com/library/llava:7b-v1.6-vicuna-q2_K)             |

#### LLaVA Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-lva-34b-4k-fp16")
```

#### LLaVA Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### LLaVA System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. Users should test the models on their own systems to determine compatibility.

- **Memory:** Estimated 64GB RAM for high-performance models. For lower quantized models (e.g., q2, q4), lower memory configurations might suffice.
- **Storage:** Approximately 75GB free disk space for the full model. Lower quantized versions will require less storage.
- **Processor:** Modern multi-core CPU (e.g., Intel i9 or AMD Ryzen 9) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA RTX 3090 or higher).

#### LLaVA Additional Information

LLaVA models are versatile and can be used for a variety of natural language processing tasks. They are particularly well-suited for applications that require high accuracy and robustness, such as content generation, summarization, and language translation. The different quantization options allow for flexibility in balancing performance and resource usage.

### DBRX

**Function Calling**: No
**Multimodal Capabilities**: No

#### DBRX Description

DBRX is a transformer-based decoder-only large language model (LLM) that was trained using next-token prediction. It uses a fine-grained mixture-of-experts (MoE) architecture with 132B total parameters of which 36B parameters are active on any input. It was pre-trained on 12T tokens of text and code data.

It is an especially capable code model, surpassing specialized models like CodeLLaMA-70B on programming, in addition to its strength as a general-purpose LLM.

DBRX excels in several key areas:

- **Programming and Mathematics**: It scores the highest among open models on benchmarks like HumanEval and GSM8k, outperforming models specifically designed for coding tasks.
- **General Knowledge and Reasoning**: DBRX surpasses leading models on benchmarks like MMLU, HellaSwag, and WinoGrande, making it highly effective for general-purpose question answering and logical reasoning tasks.
- **Composite Benchmarks**: It leads on the Hugging Face Open LLM Leaderboard and Databricks Gauntlet, showcasing its all-around excellence across various domains.

For more detailed information, visit [DBRX Ollama model card](https://ollama.com/library/dbrx) | [Databrix](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) | [HuggingFace Model Card](https://huggingface.co/databricks/dbrx-instruct)

#### DBRX Variations

| Model Reference         | Parameters | Context Window | Quantization | Download Size | Model Card                                                      |
|-------------------------|-----------:|---------------:|-------------:|--------------:|-----------------------------------------------------------------|
| o-dbrx-132b-32k-fp16    |       132b |           32k  |         fp16 |         263GB | [DBRX 132b 32k fp16](https://ollama.com/library/dbrx:132b-instruct-fp16)            |
| o-dbrx-132b-32k-q8      |       132b |           32k  |           q8 |         140GB | [DBRX 132b 32k q8](https://ollama.com/library/dbrx:132b-instruct-q8_0)              |
| o-dbrx-132b-32k-q4      |       132b |           32k  |           q4 |          74GB | [DBRX 132b 32k q4](https://ollama.com/library/dbrx:132b-instruct-q4_0)              |
| o-dbrx-132b-32k-q2      |       132b |           32k  |           q2 |          48GB | [DBRX 132b 32k q2](https://ollama.com/library/dbrx:132b-instruct-q2_K)              |

#### DBRX Usage Example

```python
from model_factory import get_model

factory = ModelFactory()
llm = factory.get_model(model="o-dbrx-132b-32k-fp16")
```

#### DBRX Performance and Benchmarks

You_can_use_this_section_to_add_your_notes_as_you_test_and_get_expereince_with_the_model_and_its_variations_and_if_you_share_them_in_the_discord_server_I_can_add_them_to_the_github_repo

#### DBRX System Requirements

Very little information is available about system requirements. The following are best estimates based on available data and typical usage. Note that actual requirements may vary depending on the load from other applications running on your machine. Users should test the models on their own systems to determine compatibility.

- **Memory:** Estimated 128GB RAM for high-performance models. For lower quantized models (e.g., q2, q4), lower memory configurations might suffice.
- **Storage:** Approximately 263GB free disk space for the full model. Lower quantized versions will require less storage.
- **Processor:** Modern multi-core CPU (e.g., Intel Xeon or AMD EPYC) recommended. Models can also benefit from GPU acceleration (e.g., NVIDIA A100 or higher).

#### DBRX Additional Information

DBRX models are well-suited for high-end applications requiring significant computational power and precision, such as advanced data analysis, large-scale language modeling, and other resource-intensive AI tasks. The different quantization options allow for flexibility in balancing performance and resource usage.
