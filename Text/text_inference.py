import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from Text import config
except ImportError:
    import config

class TextEmotionPredictor:
    def __init__(self, model_dir=config.MODEL_SAVE_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        
        if not os.path.exists(self.model_dir) or len(os.listdir(self.model_dir)) == 0:
            print(f"WARNING: No trained model found at {self.model_dir}. Using untrained base model for demonstration.")
            self.model_name = config.MODEL_NAME
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=config.NUM_CLASSES,
                id2label=config.ID_TO_LABEL,
                label2id=config.LABEL_TO_ID
            ).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device)
            print(f"Loaded trained model from {self.model_dir}")
            
        self.model.eval()

    def predict_probabilities(self, text):
        """
        Returns the output prediction vector matching the exact 7 emotion classes format
        EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        """
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=config.MAX_LENGTH,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        prob_list = [float(probs[i]) for i in range(config.NUM_CLASSES)]
        prob_dict = {config.ID_TO_LABEL[i]: float(probs[i]) for i in range(config.NUM_CLASSES)}
        
        return prob_list, prob_dict

if __name__ == "__main__":
    predictor = TextEmotionPredictor()
    #  ("angry", "How dare you say that to me?! I will never forgive you!"),
    #     ("disgust", "Ugh, this place smells like rotting garbage, it's revolting."),
    #     ("fear", "Wait, did you hear those footsteps? What if someone is in the house?"),
    #     ("happy", "I just got the job! I am so incredibly excited right now!"),
    #     ("neutral", "I am just going to the store to pick up some eggs and milk."),
    #     ("sad", "It hurts so much. I just feel completely empty and exhausted inside."),
    #     ("surprise", "Woah, I had absolutely no idea you were coming! What a shock!")

    # text = "Wait, did you hear those footsteps? What if someone is in the house?"
    text = "I am so terrified and clueless about what is going on."
    
    print(f"\nEvaluating text -> '{text}'\n")
    prob_list, prob_dict = predictor.predict_probabilities(text)
    
    print("Output Probability Vector Shape matches:", len(config.EMOTIONS))
    print("Vector:", [f"{p:.4f}" for p in prob_list])
    print("\nDetailed breakdown:")
    for k, v in prob_dict.items():
        print(f"{k}: {v:.4f}")
