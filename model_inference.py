import os 
import numpy as np 
import torch 

from peft import PeftModel
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast

class Model_inference:

    def __init__(self,
        base_model_path: str | os.PathLike, 
        lora_model_path: str | os.PathLike,
        device: str='auto') -> None:
        """ Обертка для работы с моделями инференса

        Args:
            base_model_path (str | os.PathLike): Путь к репозиторию базовой модели
                                                или имя репо на huggingface
            lora_model_path (str | os.PathLike): Путь к репозиторию адаптера модели
                                                или имя репо на huggingface 
            device (str, optional): Куда сгружать модель. Defaults to 'auto'.
        """

        base_model = Gemma2ForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=50,
            torch_dtype=torch.float16,
            device_map=device,
            use_cache=False
        )
        self.model = PeftModel.from_pretrained(base_model, lora_model_path)
        self.tokenizer = GemmaTokenizerFast.from_pretrained(lora_model_path)
        self.tokenizer.add_eos_token = True  # We'll add <eos> at the end
        self.tokenizer.padding_side = "right"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def inference(self, review:str) -> np.array:
        """ Прогоняет подготовленный отзыв через модель

        Args:
            review (str): Отзыв

        Returns:
            np.array: Массив вероятностей принадлежностей к каждому классу 
        """
    
        self.model.eval()
        tokenized_input = self.tokenizer(review, padding=True, truncation=True, return_tensors="pt")
        tokenized_input = {k: v.to(self.device) for k, v in tokenized_input.items()}

        with torch.no_grad():
            outputs = self.model(**tokenized_input).logits.cpu()
            probs = torch.sigmoid(outputs).numpy()
        
        # Минутка гениального кода...
        for i in np.ndindex(probs.shape[0]):
            for t in (0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2):
                classes = np.where(probs[i] > t)[0]
                if len(classes) != 0:
                    break
            if len(classes) == 0:
                classes = [19]

        return probs, classes


