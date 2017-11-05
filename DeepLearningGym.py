import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

# Learning rate value
LR = 1e-3

# Specify the gym you want to use for the RL environment
env = gym.make('CartPole-v0')
env.reset()

# Aim high (Good score is considered within the 200 range)
goal_steps = 500
score_requirement = 50
initial_games = 10000


def random_game_initializer():
    '''Randomly plays x amount of games'''
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            # This will visualize the environment (expensive)
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                break


def initial_population():
    '''Populate the initial training data'''
    training_data = []
    scores = []
    accepted_scores = []

    # Loop through the number of games we want to play
    for _ in range(initial_games):
        score = 0

        # Store all movements into the game_memory
        game_memory = []
        previous_observation = []

        # This is the actual game that is happening
        for _ in range(goal_steps):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)

            if (len(previous_observation) > 0):
                game_memory.append([previous_observation, action])

            previous_observation = observation
            score += reward
            if done:
                break

        # Want to log the good episode data into memory buffer
        if score >= score_requirement:
            accepted_scores.append(score)

            # Convert to one-hot vector
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score: ', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


def train_model(training_data, model=False):
    '''Train the neural network model using the training data'''

    # If you saved a previous moddel, you can load it into the function
    X = np.array([i[0] for i in training_data]).reshape(-1,
                                                        len(training_data[0][0]),
                                                        1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500,
              show_metric=True, run_id='openaistuff')

    return model


def neural_network_model(input_size):
    '''Neural Network Architecture'''
    network = input_data(shape=[None, input_size, 1], name='input')

    # Define 5 fully connected network layers with 128, 256, 512, 256, 128
    # neuron structure for each layer respectively.
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    # 2 would represent the number of output actions
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')
    return model


training_data = initial_population()
model = train_model(training_data)
scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()

    for _ in range(goal_steps):
        env.render()
        if (len(prev_obs) == 0):
            action = random.randrange(0, 2)

        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,
                                                              len(prev_obs),
                                                              1))[0])
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward

        if done:
            break
    scores.append(score)

    print('Average score', sum(scores) / len(scores))
    print('Choice 1: {}, Choice 0 {}'.format(choices.count(1) / len(choices),
                                             choices.count(0) / len(choices)))