import time
from typing import Dict, List, Optional, Union, Type
import httpx
from pydantic import BaseModel

from edenai_apis.utils.exception import ProviderException
from edenai_apis.features.llm.llm_interface import LlmInterface
from edenai_apis.features.llm.chat.chat_dataclass import (
    ChatDataClass,
    ChatCompletionChoice,
    ChatCompletionUsage,
    UsageTokensDetails,
    ChatMessage,
    ChatRole,
)

class TenstorrentLlmApi(LlmInterface):
    def llm__chat(
        self,
        messages: List = [],
        model: Optional[str] = None,
        # Optional OpenAI-like params:
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        stop: Optional[str] = None,
        stop_sequences: Optional[any] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        # new params
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        seed: Optional[int] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        deployment_id=None,
        extra_headers: Optional[dict] = None,
        # soon to be deprecated
        functions: Optional[List] = None,
        function_call: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        model_list: Optional[list] = None,
        drop_invalid_params: bool = True,
        user: str | None = None,
        # catch-all for extra params
        **kwargs,
    ) -> ChatDataClass:
        params = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if drop_invalid_params:
            params = {k: v for k, v in params.items() if v is not None}

        try:
            if stream:
                content_chunks = []
                for chunk in self.client.chat.completions.create(**params):
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta and hasattr(delta, "content"):
                            content_chunks.append(delta.content)

                final_content = "".join(content_chunks)

                # Because it’s streaming, usage stats are only returned in the end.
                # So we create placeholders for all required fields:
                zero_details = UsageTokensDetails(
                    audio_tokens=0,
                    cached_tokens=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    accepted_prediction_tokens=0,
                    reasoning_tokens=0,
                    rejected_prediction_tokens=0,
                )
                usage_obj = ChatCompletionUsage(
                    completion_tokens_details=zero_details,
                    prompt_tokens_details=zero_details,
                    total_tokens=0
                )

                # Build a ChatDataClass with placeholders
                chat_data = ChatDataClass(
                    id="stream-temp-id",
                    object="chat.completion",
                    created=int(time.time()),
                    model=model or "unknown-model",
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatMessage(
                                role=ChatRole.ASSISTANT,
                                content=final_content
                            ),
                            finish_reason="stream_finished"
                        )
                    ],
                    usage=usage_obj
                )

            else:
                response = self.client.chat.completions.create(**params)

                # Extract usage info
                raw_usage = getattr(response, "usage", None)
                if raw_usage:
                    prompt_tokens = getattr(raw_usage, "prompt_tokens", 0)
                    completion_tokens = getattr(raw_usage, "completion_tokens", 0)
                    total_tokens = getattr(raw_usage, "total_tokens", 0)
                else:
                    # If usage is missing, provide zeros
                    prompt_tokens = 0
                    completion_tokens = 0
                    total_tokens = 0

                completion_details = UsageTokensDetails(
                    audio_tokens=0,
                    cached_tokens=0,
                    prompt_tokens=0,
                    completion_tokens=completion_tokens,
                    total_tokens=0,  # or your own logic
                    accepted_prediction_tokens=0,
                    reasoning_tokens=0,
                    rejected_prediction_tokens=0,
                )

                prompt_details = UsageTokensDetails(
                    audio_tokens=0,
                    cached_tokens=0,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    total_tokens=0,  # or your own logic
                    accepted_prediction_tokens=0,
                    reasoning_tokens=0,
                    rejected_prediction_tokens=0,
                )

                usage_obj = ChatCompletionUsage(
                    completion_tokens_details=completion_details,
                    prompt_tokens_details=prompt_details,
                    total_tokens=total_tokens
                )

                # Build choices
                choices_list = []
                for choice in getattr(response, "choices", []):
                    # For each choice, we assume .index, .message, .finish_reason
                    choice_index = getattr(choice, "index", 0)
                    choice_message = getattr(choice, "message", None)
                    finish_reason = getattr(choice, "finish_reason", "stop")

                    # If choice_message is an object with .role and .content:
                    role_str = getattr(choice_message, "role", "assistant")
                    content_str = getattr(choice_message, "content", "")

                    msg_obj = ChatMessage(
                        role=ChatRole(role_str),  # must match ChatRole's Enum
                        content=content_str
                    )

                    choices_list.append(
                        ChatCompletionChoice(
                            index=choice_index,
                            message=msg_obj,
                            finish_reason=finish_reason
                        )
                    )

                chat_data = ChatDataClass(
                    id=getattr(response, "id", "no-id-provided"),
                    object=getattr(response, "object", "chat.completion"),
                    created=getattr(response, "created", int(time.time())),
                    model=getattr(response, "model", model or "unknown-model"),
                    choices=choices_list,
                    usage=usage_obj
                )

            return chat_data

        except Exception as e:
            raise ProviderException(f"TenstorrentLlmApi error: {str(e)}") from e
