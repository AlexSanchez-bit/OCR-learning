from typing import Optional
import matplotlib.pyplot as plt
from ml_lib.trainer.history import HistoryItem,History
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class RealTimeDataObserver:
    def __init__(self):
        # los and accuracy grid
        fig, ax = plt.subplots(nrows=2,ncols=2)
        ax[0,0].set_title('training loss')
        ax[0,1].set_title('training accuracy')
        ax[1,0].set_title('validation loss')
        ax[1,1].set_title('validation accuracy')

        self.figs=fig
        self.axs=ax


        train_loss_line, = self.axs[0,0].plot([],[])
        validation_loss_line, = self.axs[1,0].plot([],[])

        train_accuracy_line, = self.axs[0,1].plot([],[])
        validation_accuracy_line, = self.axs[1,1].plot([],[])

        self.loss_line=[train_loss_line,validation_loss_line]
        self.accuracy_line=[train_accuracy_line,validation_accuracy_line]

    def confussion_matrix(self,history:History,classes,classes_labels):
        predictions=[]
        labels=[]
        fig, ax = plt.subplots(figsize=(12, 10))
        for hi in filter(lambda h:h.mode=='validation',history.get_history()):
            for classified in  hi.model_outputs:
                predictions.extend(classified.argmax(dim=1).cpu().numpy())
            for label in  hi.expected_values:
                labels.extend(label.cpu().numpy())

        cm = confusion_matrix(predictions, labels, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes_labels)
        disp.plot(ax=ax)
        plt.show()

    def observe(self,data:HistoryItem):
        index=0
        now = time.time()
        if data.mode=='validation':
            index=1

        x = list(self.loss_line[index].get_xdata())
        y = list(self.loss_line[index].get_ydata())

        x.append(now)
        y.append(data.loss)

        self.loss_line[index].set_data(x, y)

        x = list(self.accuracy_line[index].get_xdata())
        y = list(self.accuracy_line[index].get_ydata())

        x.append(now)
        y.append(data.accuracy)

        self.accuracy_line[index].set_data(x, y)

        self.axs[index,0].relim()
        self.axs[index,1].relim()

        self.axs[index,0].autoscale_view()
        self.axs[index,1].autoscale_view()

        self.figs.canvas.draw_idle()


    def start_loss_graph(self):
        plt.tight_layout(pad=2.0)
        plt.show()




