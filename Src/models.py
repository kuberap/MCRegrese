import tensorflow as tf
# furt je to oznacene jako nezname, ale funguje to.
keras = tf.keras
from keras.layers import Dense, InputLayer, Dropout
from keras import layers, models


def get_model(in_dim, out_dim, units=[32,32,32], dropouts=[0.1,0.1,0.1], activations=['relu', 'relu', 'relu']):
  assert len(units)==len(dropouts), f"Different length of units and dropouts {len(units)}:{len(dropouts)}"
  assert len(units) == len(activations), f"Different length of units and activations {len(units)}:{len(activations)}"

  model = models.Sequential()
  model.add(InputLayer(input_shape = (in_dim,)))
  for u,d,a in zip(units, dropouts, activations):
    model.add(Dense(units = u, activation = a))
    model.add(Dropout(d))
  model.add(Dense(units=out_dim, activation="tanh"))
  return model


