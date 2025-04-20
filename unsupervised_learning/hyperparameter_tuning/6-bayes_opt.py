#!/usr/bin/env python3
"""Performs Bayesian optimization on deep neural network model"""

import numpy as np
from tensorflow import keras as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from GPyOpt.methods import BayesianOptimization
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import GPyOpt


data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_val, y_train, y_val = train_test_split(
	X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

def create_model(params):
	"""Builds the neural network model"""
	learning_rate, units, dropout, epochs, batch_size = params
	model = K.Sequential([
		K.layers.Flatten(input_shape=(30,)),
		K.layers.Dense(units, activation='relu'),
		K.layers.Dropout(dropout),
		K.layers.Dense(1, activation='sigmoid')
	])
	model.compile(
		loss='binary_crossentropy',
		optimizer=K.optimizers.Adam(learning_rate=learning_rate),
		metrics=['accuracy']
	)
	return model

def optimize_model(params):
	"""Defines hyperparameters and trains the model"""
	learning_rate, units, dropout, epochs, batch_size = params[0]
	units = int(units)
	epochs = int(epochs)
	batch_size = int(batch_size)
	model = create_model([learning_rate, units, dropout, epochs, batch_size])
	early_stopping = K.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
	checkpoint_path = (
        f"model_lr{learning_rate:.5f}_units{int(units)}_dropout{dropout:.2f}_epochs{int(epochs)}_batch{int(batch_size)}.h5"
    )
	checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=0
)
	history = model.fit(
		X_train, y_train,
		validation_data=(X_val, y_val),
		epochs=epochs,
		batch_size=batch_size,
		callbacks=[early_stopping, checkpoint],
		verbose=0
	)
	val_loss = history.history['val_loss'][-1]

	with open("bayes_opt.txt", "a") as f:
		f.write(f"learning rate: {learning_rate}, units: {units}, dropout rate: {dropout}, "
        f"epochs : {epochs}, batch_size: {batch_size}, Validation loss: {val_loss}\n"
		)
	return val_loss

bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
    {'name': 'units', 'type': 'discrete', 'domain': (16, 32, 64, 128)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'epochs', 'type': 'discrete', 'domain': (10, 15, 20, 25)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64)}
]

def bayesian_optimization():
    """Performs Bayesian optimisation"""
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=optimize_model,
        domain=bounds,
        acquisition_type='EI',
    )
    optimizer.run_optimization(max_iter=30)
    optimizer.plot_convergence()
    with open("bayes_opt.txt", "a") as f:
    	f.write(
		 f"Best parameters: {optimizer.X[np.argmin(optimizer.Y)]}, "
		 f"Best validation loss: {np.min(optimizer.Y)}\n"
		)
    plt.savefig('convergence.png')
    plt.show()
    return optimizer

if __name__ == "__main__":
    bayesian_optimization()
