import time
import httpx
import json
from typing import Dict, List, Optional, Union, Type, Generator

from pydantic import BaseModel

from edenai_apis.utils.exception import ProviderException
from edenai_apis.features.llm.llm_interface import LlmInterface
from edenai_apis.utils.types import ResponseType

# For standard chat completion structure:
from edenai_apis.features.llm.chat.chat_dataclass import (
    ChatDataClass,            
    ChatCompletionChoice,    
    ChatCompletionUsage,     
    UsageTokensDetails,       
    ChatMessage,              
    ChatRole,                 
)
# For streaming responses:
from edenai_apis.features.text.chat.chat_dataclass import (
    ChatStreamResponse,
    StreamChat,
)

class TenstorrentLlmApi(LlmInterface):
    provider_name = "tenstorrent"

    def __init__(self, client):
        self.client = client

    def llm__chat(
        self,
        messages: List[Dict] = [],
        model: Optional[str] = None,
        # OpenAI-like params:
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
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:

        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        # Drop any None fields if desired
        if drop_invalid_params:
            payload = {k: v for k, v in payload.items() if v is not None}
        print(f"[TenstorrentLlmApi] Final payload after dropping invalid: {payload}")

        try:
            response = self.client.chat.completions.create(**payload)
        except Exception as exc:
            raise ProviderException(str(exc))

        if not stream:
            raw = response.to_dict() if hasattr(response, "to_dict") else response

            # Extract usage, or build empty usage if missing
            usage_dict = raw.get("usage", {})

            # Build the usage details safely
            def make_usage_details(part: str) -> UsageTokensDetails:
                return UsageTokensDetails(
                    audio_tokens=0,
                    cached_tokens=0,
                    prompt_tokens=usage_dict.get(part, 0),
                    completion_tokens=0,
                    total_tokens=0,
                    accepted_prediction_tokens=0,
                    reasoning_tokens=0,
                    rejected_prediction_tokens=0,
                )

            usage = ChatCompletionUsage(
                # if total_tokens is not known or in usage_dict, fallback to 0
                total_tokens=usage_dict.get("total_tokens", 0),
                prompt_tokens_details=make_usage_details("prompt_tokens"),
                completion_tokens_details=make_usage_details("completion_tokens"),
            )

            # Build choices
            raw_choices = raw.get("choices", [])
            choices_list = []
            for i, choice in enumerate(raw_choices):
                msg = choice.get("message", {})
                role_str = msg.get("role", "assistant")
                try:
                    role_enum = ChatRole(role_str)
                except ValueError:
                    role_enum = ChatRole.ASSISTANT

                message_obj = ChatMessage(
                    role=role_enum,
                    content=msg.get("content"),
                    tool_calls=msg.get("tool_calls", None),
                )

                finish_reason = choice.get("finish_reason", "stop")
                choice_obj = ChatCompletionChoice(
                    index=choice.get("index", i),
                    message=message_obj,
                    finish_reason=finish_reason,
                )
                choices_list.append(choice_obj)

            standardized_response = ChatDataClass(
                id=raw.get("id", "dummy-id"),
                object=raw.get("object", "chat.completion"),
                created=raw.get("created", int(time.time())),
                model=raw.get("model", "dummy-model"),
                usage=usage,
                choices=choices_list,
            )

            return ResponseType[ChatDataClass](
                original_response=raw,
                standardized_response=standardized_response,
                usage=usage,
            )

        else:
            def stream_generator() -> Generator[ChatStreamResponse, None, None]:
                for idx, chunk in enumerate(response):
                    if not chunk:
                        continue
                    chunk_dict = chunk.to_dict() if hasattr(chunk, "to_dict") else chunk

                    choice = chunk_dict.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")
                    is_done = (finish_reason in (None, "stop"))
                    text = delta.get("content", "")

                    yield ChatStreamResponse(
                        text=text,
                        blocked=not is_done,
                        provider=self.provider_name,
                    )

            return ResponseType[StreamChat](
                original_response=None,
                standardized_response=StreamChat(stream=stream_generator()),
            )
