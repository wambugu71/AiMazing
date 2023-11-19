#The script has been made by wambugu kinyua.
#access the huggingchat api
#its a custom wrapper for a langchain llm
import time
from typing import Any, List, Mapping, Optional
from hugchat import hugchat
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class AiAmaizing_llm(LLM):

    """HuggingChat LLM wrapper."""

    chatbot : Optional[hugchat.ChatBot] = None

    email : Optional[str] = None
    psw : Optional[str] = None
    web_search: Optional[bool]= False
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 0.65
    repetition_penalty: Optional[float] = 1.2
    top_k: Optional[int]= 50
    truncate: Optional[int] = 1000
    watermark: Optional[bool] = False
    max_new_tokens: Optional[int] = 4096
    stop: Optional[list] = ["</s>"]
    return_full_text: Optional[bool] = False,
  #  stream: Optional[bool] = False,
    _stream_yield_all: Optional[bool] = False
    use_cache: Optional[bool] = False
    is_retry: Optional[bool] = False
    retry_count: Optional[int] = 5
    chatbot : Optional[hugchat.ChatBot] = None
  #  conversation: Optional[conversation] = None
    cookie_path : Optional[str] = None



    @property
    def _llm_type(self) -> str:
        return "Custom llm for llama2 HuggingChat api. Made by wambugu kinyua 🤠🥳"


    def create_chatbot(self) -> None:
        if not any([self.email, self.psw, self.cookie_path]):
            raise ValueError("email, psw, or cookie_path is required.")

        try:
            if self.email and self.psw:

                from hugchat.login import Login

                sign = Login(self.email, self.psw)
                cookies = sign.login()

            else:
                cookies = self.cookie_path and hugchat.ChatBot(cookie_path=self.cookie_path)

            self.chatbot = cookies.get_dict() and hugchat.ChatBot(cookies=cookies.get_dict())
        except Exception as e:
            raise ValueError("Login failed. Please check your email and password " + str(e))




    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop:
            raise ValueError("stop kwargs are not permitted.")

        self.create_chatbot() if not self.chatbot else None

        try:
#You can change the model
# index 0 for llama2 70b, 1 fo codellama, 2 for falcon 180b , 3 for Mistral 7b Ai
            self.chatbot.switch_llm(3) #falcon 180b
            self.chatbot.new_conversation(switch_to=True, system_prompt = "From now  You  are  Ai-Squad-Amaizing Ai group, You are a  great helper. Output  in this form '{Question}: Answer: {Answer  to the  question}'")
            resp = self.chatbot.query(
                prompt, temperature=self.temperature,
        top_p= self.top_p,
        repetition_penalty = self.repetition_penalty,
        top_k = self.top_k,
        truncate= self.truncate,
        watermark= self.watermark,
        max_new_tokens = self.max_new_tokens,
        stop = self.stop,
        return_full_text = self.return_full_text,
       # stream = self.stream,
        _stream_yield_all = self._stream_yield_all,
        use_cache = self.use_cache,
        is_retry = self.is_retry,
        retry_count = self.retry_count,

        )
            return resp['text']

        except Exception as e:
            raise ValueError("ChatBot failed, please check your parameters. " + str(e))

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        params = {"web_search" :self.web_search,
"temperature": self.temperature,
"top_p": self.top_p,"repetition_penalty" : self.repetition_penalty,
"top_k" : self.top_k,"truncate" :self.truncate,"watermark" : self.watermark,"max_new_tokens" : self.max_new_tokens,
"stop" : self.stop,"return_full_text" : self.return_full_text,"_stream_yield_all" : self._stream_yield_all,
"use_cache" : self.use_cache,"is_retry"  : self.is_retry, "retry_count" : self.retry_count }

        return params
