import numpy
import pandas
from sklearn.metrics import mean_absolute_error


preds = list()
for i in range(0, 5):
    preds.append(numpy.array(pandas.read_excel('save/tdhhp/preds_' + str(i) + '.xlsx', header=None)))
preds = numpy.vstack(preds)

print(mean_absolute_error(preds[:, 0], preds[:, 1]))
