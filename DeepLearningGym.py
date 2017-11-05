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


def random_game_initializer(simulation_count=10):
    '''
        Randomly plays x amount of games
        Used to simulate what a failing robot play looks like
    '''
    for episode in range(simulation_count):
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

            # Basically some epsilon greedy approach to picking left-right move
            action = random.randrange(0, 2)

            # Gather all the information post movement after taking action
            observation, reward, done, info = env.step(action)

            # Store that observation information into the memory buffer
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

                # Training data will consist of feeding in good results
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

    # If you saved a previous model, you can load it into the function
    X = np.array([i[0] for i in training_data]).reshape(-1,
                                                        len(training_data[0][0]),
                                                        1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500,
              show_metric=True, run_id='openaistuff')

    return model


def neural_network_model(input_size):
    '''Neural Network Architecture'''
    network = input_data(shape=[None, input_size, 1], name='input')

    # Define 5 fully connected network layers with 128, 256, 512, 256, 128
    # neuron structure for each layer respectively.
    network = fully_connected(network, 64, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 32, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 64, activation='relu')
    network = dropout(network, 0.8)

    # 2 would represent the number of output actions
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')
    return model


visualize_game_play = raw_input('Do you want to see sample gameplay?: 0/1')
if (visualize_game_play == '1'):
    game_count = raw_input('Input number of simulations to visualize: ')
    print('Running simulations')
    random_game_initializer(int(game_count))

training_data = initial_population()
model = train_model(training_data)
scores = []
choices = []
viz = 0
if (raw_input("Do you want to visualize outcome? 0/1 ") == '1'):
    viz = 1

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()

    for _ in range(goal_steps):
        # Uncomment for game graphics simulation
        if viz == 1:
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

    if (sum(scores) / len(scores) >= 200):
        model.save('AWESOMEMODEL.model')
        print('Model saved successfully!')

    print('Choice 1: {}, Choice 0 {}'.format(choices.count(1) / len(choices),
                                             choices.count(0) / len(choices)))
