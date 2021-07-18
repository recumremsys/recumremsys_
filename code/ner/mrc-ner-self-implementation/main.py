import numpy as np
import tensorflow as tf
from bert_layer import BertLayer
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K
from data_generator import DataGenerator


#this function will give span matching predictions
def span_matrix_func(tensor):
    embeddings = tensor
    start_expand = K.tile(K.expand_dims(embeddings, 2), [1, 1, 128, 1])
    end_expand = K.tile(K.expand_dims(embeddings, 1), [1, 128, 1, 1])

    span_matrix = K.concatenate([start_expand, end_expand], 3)         
    
    return span_matrix

def build_function(max_seq_length):
    input_ids = layers.Input(shape = (max_seq_length), name = "input_ids")
    input_mask = layers.Input(shape = (max_seq_length), name = "input_mask")
    segment_ids = layers.Input(shape = (max_seq_length), name = "segment_ids")
    bert_input = [input_ids, input_mask, segment_ids]

    bert_output = BertLayer()(bert_input)
    start_predictions_layer = layers.Dense(2, activation='softmax', name = "start_prediction_layer")
    end_predictions_layer = layers.Dense(2, activation='softmax', name = "end_prediction_layer")

    start_logits = layers.TimeDistributed(start_predictions_layer, name = "start_logits")(bert_output)
    end_logits = layers.TimeDistributed(end_predictions_layer, name = "end_logits")(bert_output)

    span_matrix = layers.Lambda(span_matrix_func, name = "span_matrix")(bert_output)
    span_logits = layers.Conv2D(
                                1,
                                1,
                                input_shape = (max_seq_length, max_seq_length, 2*768),
                                activation="relu",
                                name = "span_logits"
                                )(span_matrix)
    flat_span = layers.Flatten(name = "span_flat")(span_logits)
    flat_start = layers.Flatten(name = "start_flat")(start_logits)
    flat_end = layers.Flatten(name = "end_flat")(end_logits)
    
    outputs = [flat_start, flat_end, flat_span]

    model = models.Model(inputs = bert_input, outputs = outputs)

    return model


max_seq_length = 128
model = build_function(max_seq_length)

losses = {
    "span_flat" : "binary_crossentropy",
    "start_flat" : "binary_crossentropy",
    "end_flat" : "binary_crossentropy",
}
losses_weights = {
    "span_flat" : 1.0,
    "start_flat" : 1.0,
    "end_flat" : 1.0,
}

model.compile(loss = losses, loss_weights = losses_weights, optimizer="adam", metrics=["accuracy"])
print(model.summary())

tf.keras.utils.plot_model(
    model,
    to_file="model.png",
)

train_data = DataGenerator(mode="train")
validation_data = DataGenerator(mode="dev")
test_data = DataGenerator(mode="test")

model.fit(
    train_data,
    validation_data = validation_data,
    steps_per_epoch=len(train_data),
    epochs = 100,
)