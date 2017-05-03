import os
import time
import pickle
import matplotlib.pyplot as plt

class Reporter():

    def __init__(self, infos=(0, 0, 0), report_dir='./report'):
        if os.path.exists(report_dir) == False:
            os.mkdir(report_dir)
        self.report_id = "{}-{}-epoch{}-{}".format(infos[0], infos[1], infos[2], str(time.time()))
        if os.path.exists(os.path.join(report_dir, self.report_id)) == False:
            os.mkdir(os.path.join(report_dir, self.report_id))  # ディレクトリを作成する
        self.report_dir = os.path.join(report_dir, self.report_id)

    def save_history(self, history):     # 学習履歴を保存する
        with open(os.path.join(self.report_dir, "history-{}.pickle".format(self.report_id)), mode='wb') as f:
            pickle.dump(history.history, f)

    def save_model(self, model):        # モデル構造のみ（weightを含まない）の保存
        model_json = model.to_json()
        with open(os.path.join(self.report_dir, "model-{}.json".format(self.report_id)), mode='w') as f:
            f.write(model_json)

    def save_weights(self, model):      # 学習済みモデルの重みを保存する
        model.save_weights(os.path.join(self.report_dir, "weights-{}.hdf5".format(self.report_id)))

    def save_loss_plot(self, history, show=False):      # 損失の履歴をプロット
        plt.plot(history.history['loss'],"o-",label="loss",)
        plt.plot(history.history['acc'],"o-",label="acc")
        plt.plot(history.history['val_loss'],"o-",label="val_loss")
        plt.plot(history.history['val_acc'],"o-",label="val_acc")
        plt.title('loss graph')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.report_dir, "graph-loss-{}.png".format(self.report_id)))
        if show:
            plt.show()
        plt.clf()   # plotを初期化

    def report(self, model=None, history=None, show_loss_plot=False):
        self.save_history(history)  # 学習履歴を保存する
        self.save_model(model)      # モデル構造を保存する
        self.save_weights(model)    # 重みを保存する
        self.save_loss_plot(history, show=show_loss_plot)    # 損失グラフを保存する

