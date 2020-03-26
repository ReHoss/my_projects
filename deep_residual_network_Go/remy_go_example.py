import numpy as np
from tensorflow import keras
import tensorflow.keras.layers as layers
import golois
import pickle

PLANES = 8
MOVES = 361

N_FIlTERS = 108
N_HIDDEN = 4
bn_axis = 3


# Building model

keras_input = keras.Input(shape=(19, 19, PLANES), name='board')

x = layers.Conv2D(N_FIlTERS, 1, padding='same')(keras_input)
x = layers.BatchNormalization(axis=bn_axis)(x)
x = layers.Activation('relu')(x)

x_old = x

x = layers.Conv2D(N_FIlTERS, 5, padding='same')(keras_input)
x = layers.BatchNormalization(axis=bn_axis)(x)
x = layers.Activation('relu')(x)

x = layers.add([x, x_old])
x_old = x


x = layers.Conv2D(N_FIlTERS, 1, padding='same')(x)
x = layers.BatchNormalization(axis=bn_axis)(x)
x = layers.Activation('relu')(x)

x = layers.add([x, x_old])
x_old = x

for i in range(N_HIDDEN):

    x = layers.Conv2D(N_FIlTERS, 1, padding='same')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(N_FIlTERS, 3, padding='same')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    if i < (N_HIDDEN - 1):
        x = layers.Conv2D(N_FIlTERS, 3, padding='same')(x)
        x = layers.BatchNormalization(axis=bn_axis)(x)
        x = layers.Activation('relu')(x)
        x = layers.add([x, x_old])

    else:
        x = layers.Conv2D(N_FIlTERS // 2 - 10, 3, padding='same')(x)
        x = layers.BatchNormalization(axis=bn_axis)(x)
        x = layers.Activation('relu')(x)

    x_old = x


# x = layers.Conv2D(N_FIlTERS, 3, padding='same')(x)
# x = layers.BatchNormalization(axis=bn_axis)(x)
# x = layers.Activation('relu')(x)
# x = layers.add([x, x_old])

x = layers.Conv2D(N_FIlTERS // 2 - 10, 1, padding='same')(x)
x = layers.BatchNormalization(axis=bn_axis)(x)
x = layers.Activation('relu')(x)
x = layers.add([x, x_old])

policy_head = layers.Conv2D(1, 2, padding='same')(x)
policy_head = layers.BatchNormalization(axis=bn_axis)(policy_head)
policy_head = layers.Activation('relu')(policy_head)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Dense(MOVES, activation='softmax', name='policy')(
        policy_head)

# value_head = layers.Conv2D(1, 1, padding='same')(x)
# value_head = layers.BatchNormalization(axis=bn_axis)(value_head)
# value_head = layers.Activation('relu')(value_head)
value_head = layers.AveragePooling2D(padding='same')(x)
value_head = layers.AveragePooling2D()(value_head)
value_head = layers.Conv2D(1, 1, padding='same')(value_head)
value_head = layers.Flatten()(value_head)
value_head = layers.Dense(28, activation='relu')(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value')(value_head)

model = keras.Model(inputs=keras_input, outputs=[policy_head, value_head])

# End model

model.summary()

# Optimizer compilation

model.compile(optimizer=keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
              loss={'value': 'mse',
                    'policy': 'categorical_crossentropy'},
              metrics=['acc'])

model.optimizer.get_config()

N = 500000
N_BATCHES = 180
EPOCHS = 1
STR_NAME = 'RemyyGo22_'
PATH = './history_RemyyGo22_4/'


START = 5
END = 180 + 3

# Training from batch 2 to 182

input_data = np.random.randint(2, size=(N, 19, 19, PLANES))
input_data = input_data.astype('float32')

policy = np.random.randint(MOVES, size=(N,))
policy = keras.utils.to_categorical(policy)

value = np.random.randint(2, size=(N,))
value = value.astype('float32')

end = np.random.randint(2, size=(N, 19, 19, 2))
end = end.astype('float32')


for i in range(START):
    print(i)
    golois.getBatch(input_data, policy, value, end)


for i in range(START, END):
    golois.getBatch(input_data, policy, value, end)
    print(f'New sample generated !')

    keras_history = model.fit(input_data,
                              {'policy': policy, 'value': value},
                              epochs=EPOCHS,
                              batch_size=256,
                              validation_split=0)

    print(f'Training on sample {i} over.')
    print()

    str_name = STR_NAME + f'{N_BATCHES}_{EPOCHS}_batch_{i}'

    model.save(PATH + str_name + '.h5')
    with open(PATH + str_name + '.pickle', 'wb') as file:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(keras_history.history, file)