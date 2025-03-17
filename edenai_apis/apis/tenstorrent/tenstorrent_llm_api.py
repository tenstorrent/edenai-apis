import time
from typing import Dict, List, Optional, Union, Type
import httpx
from pydantic import BaseModel

from edenai_apis.utils.exception import ProviderException
from edenai_apis.features.llm.llm_interface import LlmInterface

from edenai_apis.features.text.chat import ChatDataClass, ChatMessageDataClass
from edenai_apis.utils.types import ResponseType
from edenai_apis.features.text.chat.chat_dataclass import ChatStreamResponse
from edenai_apis.features.text.chat.chat_dataclass import StreamChat
import json

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
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        params = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if drop_invalid_params:
            params = {k: v for k, v in params.items() if v is not None}



        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        try:
            response = self.client.chat.completions.create(**payload)
        except Exception as exc:
            raise ProviderException(str(exc))

        # Standardize the response
        if not stream:
            message = response.choices[0].message
            generated_text = message.content

            conversation_messages = []
            for msg in messages:
                conversation_messages.append(ChatMessageDataClass(
                    role=msg["role"],
                    message=msg["content"]
                ))

            messages_json = [m.dict() for m in conversation_messages]

            standardized_response = ChatDataClass(
                generated_text=generated_text, message=messages_json
            )

            return ResponseType[ChatDataClass](
                original_response=response.to_dict(),
                standardized_response=standardized_response,
                usage=response.usage,
            )
        else:
            stream = (
                ChatStreamResponse(
                    text=chunk.to_dict()["choices"][0]["delta"].get("content", ""),
                    blocked=not chunk.to_dict()["choices"][0].get("finish_reason") in (None, "stop"),
                    provider=self.provider_name,
                )
                for chunk in response
                if chunk
            )

            return ResponseType[StreamChat](
                original_response=None, standardized_response=StreamChat(stream=stream)
            )

