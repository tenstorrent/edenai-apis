from typing import List, Optional

import requests

from edenai_apis.features.text.keyword_extraction.keyword_extraction_dataclass import (
    KeywordExtractionDataClass,
)
from edenai_apis.features.text.named_entity_recognition.named_entity_recognition_dataclass import (
    NamedEntityRecognitionDataClass,
)
from edenai_apis.features.text.question_answer.question_answer_dataclass import (
    QuestionAnswerDataClass,
)
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import (
    SentimentAnalysisDataClass,
)
from edenai_apis.features.text.text_interface import TextInterface
from edenai_apis.features.text.topic_extraction.topic_extraction_dataclass import (
    TopicExtractionDataClass,
)
from edenai_apis.features.text import ChatDataClass, ChatMessageDataClass
from edenai_apis.features.text.chat.chat_dataclass import StreamChat, ChatStreamResponse

from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class TenstorrentTextApi(TextInterface):
    def text__keyword_extraction(
        self, language: str, text: str
    ) -> ResponseType[KeywordExtractionDataClass]:
        base_url = "https://keyword-extraction--eden-ai.workload.tenstorrent.com"
        url = f"{base_url}/predictions/keyword_extraction"
        payload = {
            "text": text,
        }

        try:
            original_response = requests.post(url, json=payload, headers=self.headers)
        except requests.exceptions.RequestException as exc:
            raise ProviderException(message=str(exc), code=500)
        if original_response.status_code != 200:
            raise ProviderException(message=original_response.text, code=original_response.status_code)

        status_code = original_response.status_code
        original_response = original_response.json()

        # Check for errors
        self.__check_for_errors(original_response, status_code)

        standardized_response = KeywordExtractionDataClass(
            items=original_response["items"]
        )
        return ResponseType[KeywordExtractionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        base_url = "https://sentiment-analysis--eden-ai.workload.tenstorrent.com"
        url = f"{base_url}/predictions/sentiment_analysis"
        payload = {
            "text": text,
        }
        try:
            original_response = requests.post(url, json=payload, headers=self.headers)
        except requests.exceptions.RequestException as exc:
            raise ProviderException(message=str(exc), code=500)
        if original_response.status_code != 200:
            raise ProviderException(message=original_response.text, code=original_response.status_code)

        status_code = original_response.status_code
        original_response = original_response.json()

        # Check for errors
        self.__check_for_errors(original_response, status_code)

        # Create output response
        confidence = float(original_response["confidence"])
        prediction = original_response["prediction"]
        standardized_response = SentimentAnalysisDataClass(
            general_sentiment=prediction,
            general_sentiment_rate=confidence,
        )

        return ResponseType[SentimentAnalysisDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__question_answer(
        self,
        texts: List[str],
        question: str,
        temperature: float,
        examples_context: str,
        examples: List[List[str]],
        model: Optional[str],
    ) -> ResponseType[QuestionAnswerDataClass]:
        base_url = "https://question-answer--eden-ai.workload.tenstorrent.com"
        url = f"{base_url}/predictions/question_answer"
        payload = {
            "text": texts[0],
            "question": question,
        }
        try:
            original_response = requests.post(url, json=payload, headers=self.headers)
        except requests.exceptions.RequestException as exc:
            raise ProviderException(message=str(exc), code=500)
        if original_response.status_code != 200:
            raise ProviderException(message=original_response.text, code=original_response.status_code)

        status_code = original_response.status_code
        original_response = original_response.json()

        # Check for errors
        self.__check_for_errors(original_response, status_code)

        standardized_response = QuestionAnswerDataClass(
            answers=[original_response["answer"]]
        )
        return ResponseType[QuestionAnswerDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__named_entity_recognition(
        self, text: str, language : str
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        base_url = "https://named-entity-recognition--eden-ai.workload.tenstorrent.com"
        url = f"{base_url}/predictions/named_entity_recognition"
        payload = {
            "text": text,
        }
        try:
            original_response = requests.post(url, json=payload, headers=self.headers)
        except requests.exceptions.RequestException as exc:
            raise ProviderException(message=str(exc), code=500)
        if original_response.status_code != 200:
            raise ProviderException(message=original_response.text, code=original_response.status_code)

        status_code = original_response.status_code
        original_response = original_response.json()

        # Check for errors
        self.__check_for_errors(original_response, status_code)

        standardized_response = NamedEntityRecognitionDataClass(
            items=original_response["items"]
        )
        return ResponseType[NamedEntityRecognitionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__topic_extraction(
        self, text: str, language : str
    ) -> ResponseType[TopicExtractionDataClass]:
        base_url = "https://topic-extraction--eden-ai.workload.tenstorrent.com"
        url = f"{base_url}/predictions/topic_extraction"
        payload = {
            "text": text,
        }
        try:
            original_response = requests.post(url, json=payload, headers=self.headers)
        except requests.exceptions.RequestException as exc:
            raise ProviderException(message=str(exc), code=500)
        if original_response.status_code != 200:
            raise ProviderException(message=original_response.text, code=original_response.status_code)

        status_code = original_response.status_code
        original_response = original_response.json()

        # Check for errors
        self.__check_for_errors(original_response, status_code)

        standardized_response = TopicExtractionDataClass(
            items=original_response["items"]
        )
        return ResponseType[TopicExtractionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
    
    def text__chat(
        self,
        text: str,
        chatbot_global_action: Optional[str],
        previous_history: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        model: str,
        stream=False,
        available_tools: Optional[List[dict]] = None,  # Kept in signature for compatibility
        tool_choice: Literal["auto", "required", "none"] = "auto",  # Kept in signature
        tool_results: Optional[List[dict]] = None,  # Kept in signature
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:

        # Ensure previous_history is a list
        previous_history = previous_history or []

        messages = []

        # Convert previous_history to OpenAI-compatible messages
        for msg in previous_history:
            messages.append(
                {
                    "role": msg.get("role"),
                    "content": msg.get("message"),
                }
            )

        # Add the user's message
        if text:
            messages.append({"role": "user", "content": text})

        # Insert system message if chatbot_global_action is present and model is not O1
        if chatbot_global_action:
            messages.insert(0, {"role": "system", "content": chatbot_global_action})

        # Prepare the payload for OpenAI
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "stream": stream,
        }

        # Call the OpenAI chat completion API
        try:
            response = self.client.chat.completions.create(**payload)
        except Exception as exc:
            raise ProviderException(str(exc))

        # Handle non-streaming responses
        if not stream:
            message = response.choices[0].message
            generated_text = message.content

            # Build standardized messages
            messages_out = [
                ChatMessageDataClass(role="user", message=text),
                ChatMessageDataClass(role="assistant", message=generated_text),
            ]
            messages_json = [m.dict() for m in messages_out]

            # Create the standardized response
            standardized_response = ChatDataClass(
                generated_text=generated_text,
                message=messages_json,
            )

            return ResponseType[ChatDataClass](
                original_response=response.to_dict(),
                standardized_response=standardized_response,
            )

        # Handle streaming responses
        else:
            stream_generator = (
                ChatStreamResponse(
                    text=chunk.to_dict()["choices"][0]["delta"].get("content", ""),
                    blocked=not chunk.to_dict()["choices"][0].get("finish_reason")
                    in (None, "stop"),
                    provider="openai",
                )
                for chunk in response
                if chunk
            )

            return ResponseType[StreamChat](
                original_response={},
                standardized_response=StreamChat(stream=stream_generator),
            )


    def __check_for_errors(self, response, status_code = None):
        if "message" in response:
            raise ProviderException(response["message"], code= status_code)
