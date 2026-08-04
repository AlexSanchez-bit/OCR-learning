from typing import List

class History:
    def __init__(self):
        self.history:List[HistoryItem]=[]
    def get_history(self):
        return self.history
    def register(self,item:HistoryItem):
        self.history.append(item)
    def __len__(self):
        return len(self.history)


class HistoryItem:
    def __init__(self,
                 mode:str,
                 loss:float,
                 accuracy:float,
                 model_outputs,
                 expected_values,
                 ):
        self.mode = mode
        self.loss = loss
        self.accuracy = accuracy
        self.model_outputs=model_outputs
        self.expected_values=expected_values
    def __str__(self):
        return f'{'{'}"mode":"{self.mode}","loss":"{self.loss},"accuracy":"{self.accuracy}"{'}'}'
    def __repr__(self):
        return f'{'{'}"mode":"{self.mode}","loss":"{self.loss},"accuracy":"{self.accuracy}"{'}'}'
