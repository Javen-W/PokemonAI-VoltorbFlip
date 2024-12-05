import nltk
import keras
import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import TextVectorization
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

"""
LSTM model generator.
"""
EMBEDDING_DIM = 128
N_HIDDEN = 100
OPTIMIZER = 'adam'
N_CLASSES = 25
VISIBLE_DIM = 45


def get_model(model_path=None):
    if model_path:
        # load existing model
        model = keras.models.load_model(model_path)
    else:
        # create new model
        model = Sequential()
        embeddings = Embedding(
            input_dim=VISIBLE_DIM,
            output_dim=EMBEDDING_DIM,
        )
        model.add(embeddings)
        model.add(LSTM(N_HIDDEN, return_sequences=False))
        model.add(Dense(N_CLASSES))  # , activation='softmax'
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=OPTIMIZER,
            metrics=['accuracy']
        )
    return model


# Main function
def main():
    # Load the CSV files
    visible_states = pd.read_csv('./training_data/visible_states.csv').drop(columns=["state_index"])
    hidden_states = pd.read_csv('./training_data/hidden_states.csv').filter(like="tiles_")
    print(visible_states.shape)
    print(hidden_states.shape)

    # prepare datasets
    n_samples = len(visible_states.values) - 1
    X = visible_states.values[:n_samples]
    y = np.zeros(n_samples)
    for i in range(n_samples):
        diff = (visible_states.iloc[i + 1] - visible_states.iloc[i] >= 1)
        next_tile = int(str(list(visible_states.iloc[i].index[diff]).pop()).split("_")[1])
        y[i] = next_tile
    print(X.shape)
    print(y.shape)

    """
    Train and evaluate model.
    """
    # create new model
    model = get_model()

    # create data splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=777,
    )

    # train the model
    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=64
    )

    # final evaluation of the model
    scores = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )
    accuracy = scores[1]

    # report results
    print("Accuracy: %.2f%%" % (accuracy * 100))

    # save model
    model.save("./weights/hidden_lstm.keras")


# Entry point
if __name__ == "__main__":
    main()
