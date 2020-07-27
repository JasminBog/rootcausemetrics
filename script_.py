import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("/home/matilda/PycharmProjects/RCA_metrics /data_paper_attention_traces")
# plt.hist(data.iloc[:, [3]])
# plt.show()
lista_lstm = np.array([0.5, 2, 3.5, 7, 8.5, 10, 13.5, 15, 16.5])
lista_attention = lista_lstm+ 0.5

plt.bar(lista_lstm, data.iloc[:, 4], width=0.5, label="LSTM", color="red")
plt.bar(lista_attention, data.iloc[:, 5], width=0.5, label="ATTENTION", color="blue")
#plt.xticks(lista_lstm, ["long_prec", "long_rec", "long_f1", "short_prec", "short_rec", "short_f1", "long_short_prec", "long_short_rec", "long_short_f1"], rotation=90)
plt.xticks(lista_lstm, ["precision", "recall", "F1", "precision", "recall", "F1", "precision", "recall", "F1"], rotation=90, fontsize=12)
plt.text(2.25, 1, s="LONG" )
plt.text(8.25, 1, s="SHORT" )
plt.text(14.25, 1, s="LONG_SHORT")
plt.legend()
plt.savefig("/home/matilda/PycharmProjects/RCA_metrics /2nd.pdf", bbox_inches="tight")
