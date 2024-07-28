import pygame
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import sys
import pickle
import pathlib


# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Flappy Bird')

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

score = 0

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Bird settings
#pygame.Rect(left, top, width, height)
bird = pygame.Rect(100, 300, 30, 30)
bird_velocity = 0
gravity = 0.5

# Pipe settings
pipe_width = 50
pipe_gap = 150
pipe_velocity = -3

# Load assets
try:
    bird_image = pygame.image.load(f'{pathlib.Path(__file__).parent.resolve()}/Asset/bird.png')
    bird_image = pygame.transform.scale(bird_image, (45, 40))  # Resize bird to 30x30 pixels
    pipe_image = pygame.image.load(f'{pathlib.Path(__file__).parent.resolve()}/Asset/pipe_3.png')
    pipe_image_flip = pygame.image.load(f'{pathlib.Path(__file__).parent.resolve()}/Asset/pipe_4.png')

    # Load and scale the background image
    background_image = pygame.image.load(f'{pathlib.Path(__file__).parent.resolve()}/Asset/background_4.jpg')
    background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))

    jump_sound = pygame.mixer.Sound(f'{pathlib.Path(__file__).parent.resolve()}/Raw/jump.mp3')
    collision_sound = pygame.mixer.Sound(f'{pathlib.Path(__file__).parent.resolve()}/Raw/collision.mp3')

except pygame.error as e:
    print(f"Error loading assets: {e}")
    pygame.quit()
    sys.exit()

# Create a neural network model
model = Sequential([
    Dense(24, activation='relu', input_shape=(8,)),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

def get_state(bird, pipes, bird_velocity):
    # Normalize bird's vertical position
    bird_y = bird.y / HEIGHT
    
    # Normalize bird's vertical velocity
    bird_velocity /= 10
    
    # Normalize pipe's horizontal position
    pipe_x = pipes[0][0].x / WIDTH
    
    # Calculate the top and bottom of the gap between pipes
    top_pipe_bottom = pipes[0][1].top
    bottom_pipe_top = pipes[0][0].bottom
    
    # Normalize the top and bottom of the gap
    pipe_gap_top = top_pipe_bottom / HEIGHT
    pipe_gap_bottom = bottom_pipe_top / HEIGHT
    pipe_gap_size = (top_pipe_bottom - bottom_pipe_top) / HEIGHT
    
    # Normalize bird's dimensions
    bird_height = bird.height / HEIGHT
    bird_width = bird.width / WIDTH

    if bird.y > bottom_pipe_top:
        distance_from_gap = bird.y - bottom_pipe_top
    else:
        distance_from_gap = top_pipe_bottom - bird.y

    # Normalize the distance from the gap
    gap_height = top_pipe_bottom - bottom_pipe_top
    normalized_distance = distance_from_gap / gap_height
    
    # Return the state as a NumPy array
    return np.array([
        bird_y,                  # Bird's vertical position
        bird_velocity,           # Bird's vertical velocity
        pipe_x,                  # Distance to the nearest pipe
        pipe_gap_top,            # Top of the pipe gap
        pipe_gap_bottom,         # Bottom of the pipe gap
        pipe_gap_size,           # Size of the pipe gap
        bird_height,             # Height of the bird
        normalized_distance               # Width of the bird
    ])

def get_reward(bird, pipes):
    # Extract pipe information
    top_pipe = pipes[0][0]
    bottom_pipe = pipes[0][1]

    # Check for collision with any part of the pipes or ground
    if bird.colliderect(top_pipe) or bird.colliderect(bottom_pipe) or bird.y > HEIGHT or bird.y < 0:
        return -1  # Collision penalty

    # Calculate the gap between the top and bottom pipes
    gap_top = top_pipe.bottom
    gap_bottom = bottom_pipe.top

    # Calculate the distance from the bird's center to the bottom of the gap
    bird_center = bird.y + bird.height / 2
    distance_from_bottom = bird_center - gap_bottom

    # Normalize the distance based on the size of the gap
    gap_size = gap_bottom - gap_top
    normalized_distance = abs(distance_from_bottom) / gap_size

    # Calculate the reward
    if bird.y > gap_top and (bird.y + bird.height) < gap_bottom:
        # Reward for being within the gap and closer to the bottom
        reward = 1 - normalized_distance  # Closer to 1 means better (closer to bottom)
    else:
        # Apply same reward for being below the gap as if being within the gap
        # The further below the bottom, the higher the reward
        reward = 1 - normalized_distance  # Reward is the same as being close to the bottom

    return reward

def create_pipe():
    height = random.randint(100, 400)
    top_pipe = pygame.Rect(WIDTH, 0, pipe_width, height)
    bottom_pipe = pygame.Rect(WIDTH, height + pipe_gap, pipe_width, HEIGHT - height - pipe_gap)
    return top_pipe, bottom_pipe

def draw_bird_and_pipes(bird, pipes, score):
    screen.fill(WHITE)
    screen.blit(background_image, (0, 0))
    screen.blit(bird_image, (bird.x, bird.y))
    for pipe in pipes:
        top_pipe_image = pygame.transform.scale(pipe_image, (pipe_width, pipe[0].height))
        bottom_pipe_image = pygame.transform.scale(pipe_image_flip, (pipe_width, HEIGHT - pipe[0].height - pipe_gap))
        screen.blit(top_pipe_image, (pipe[0].x, pipe[0].y))
        screen.blit(bottom_pipe_image, (pipe[1].x, pipe[1].y))

    # Draw the score
    font = pygame.font.Font(None, 36)
    score_text = font.render(f'Score: {score}', True, BLACK)
    screen.blit(score_text, (WIDTH - score_text.get_width() - 10, 10))    
    pygame.display.flip()


def reset_game():
    global bird, bird_velocity, pipes, score
    bird = pygame.Rect(100, 300, 30, 30)
    bird_velocity = 0
    pipes = [create_pipe()]
    score = 0

def collect_training_data():
    global bird_velocity, score
    training_data = []
    reset_game()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                bird_velocity = -8
                if hasattr(pygame, 'mixer'):
                    jump_sound.play()

        # Bird movement
        bird_velocity += gravity
        bird.y += bird_velocity

        # Pipe movement
        for pipe in pipes:
            pipe[0].x += pipe_velocity
            pipe[1].x += pipe_velocity

        # Remove pipes off the screen
        if pipes[0][0].x < -pipe_width:
            pipes.pop(0)
            pipes.append(create_pipe())
            score += 1  # Increase score when passing a pip

        # Collision detection and reward calculation
        reward = get_reward(bird, pipes)
        state = get_state(bird, pipes, bird_velocity)
        training_data.append((state, reward))

        if reward == -1:
            if hasattr(pygame, 'mixer'):
                collision_sound.play()
            running = False

        # Draw the bird and pipes
        draw_bird_and_pipes(bird, pipes, score)
        clock.tick(30)

    # Save training data to a file
    return training_data
        
def train_model(model, training_data, epochs=10):
    states, rewards = zip(*training_data)
    states = np.array(states)
    rewards = np.array(rewards)
    model.fit(states, rewards, epochs=epochs)

def load_training_data():
    try:
        with open(f'{pathlib.Path(__file__).parent.resolve()}/training_data.pkl', 'rb') as f:
            training_data = pickle.load(f)
        print("Loaded existing training data.")
        return training_data
    except FileNotFoundError:
        # If file does not exist, create it with an empty list
        with open(f'{pathlib.Path(__file__).parent.resolve()}/training_data.pkl', 'wb') as f:
            pickle.dump([], f)
        print("No existing training data found. Created new file.")
        return []
    except (pickle.PickleError, EOFError) as e:
        # Handle any errors during loading
        print(f"Error loading training data: {e}")
        return []

def automatic_play():
    global bird_velocity, score
    reset_game()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        # Get state and predict action
        state = get_state(bird, pipes, bird_velocity)
        action = model.predict(state.reshape(1, 8), verbose=0)[0][0]

        #print(state,action)

        # Bird movement based on action
        if action > 0.7:
            bird_velocity = -8
            if hasattr(pygame, 'mixer'):
                jump_sound.play()

        bird_velocity += gravity
        bird.y += bird_velocity

        # Pipe movement
        for pipe in pipes:
            pipe[0].x += pipe_velocity
            pipe[1].x += pipe_velocity

        # Remove pipes off the screen
        if pipes[0][0].x < -pipe_width:
            pipes.pop(0)
            pipes.append(create_pipe())
            score += 1  # Increase score when passing a pipe

        # Collision detection
        if bird.colliderect(pipes[0][0]) or bird.colliderect(pipes[0][1]) or bird.y > HEIGHT:
            if hasattr(pygame, 'mixer'):
                collision_sound.play()
            running = False
        
        # Draw the background
        screen.blit(background_image, (0, 0))

        # Draw the bird and pipes
        draw_bird_and_pipes(bird, pipes, score)
        clock.tick(30)



def save_training_data(training_data, file_path):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(training_data, f)
        print("Training data saved successfully.")
    except Exception as e:
        print(f"Error saving training data: {e}")

# Constants
START_BUTTON_WIDTH, START_BUTTON_HEIGHT = 200, 50
START_BUTTON_X, START_BUTTON_Y = (WIDTH - START_BUTTON_WIDTH) // 2, HEIGHT // 2

# Start button rectangle
start_button_rect = pygame.Rect(START_BUTTON_X, START_BUTTON_Y, START_BUTTON_WIDTH, START_BUTTON_HEIGHT)

def menu():
    # Define colors
    BUTTON_COLOR = (55,176,233,255)
    BUTTON_COLOR_INSIDE = '#358316'
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return  # Start the game on spacebar press
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check if the mouse click is on the "Start" button
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if start_button_rect.collidepoint(mouse_x, mouse_y):
                    return  # Start the game if mouse click is on "Start" button

        # Draw the menu screen
        screen.blit(background_image, (0, 0))  # Draw background
        # Draw "Start" button
        pygame.draw.rect(screen, BUTTON_COLOR_INSIDE, start_button_rect)
        start_font = pygame.font.Font(None, 36)
        start_text = start_font.render("Start", True, WHITE)
        start_text_rect = start_text.get_rect(center=start_button_rect.center)
        screen.blit(start_text, start_text_rect)

        pygame.display.flip()
        clock.tick(30)

def main():
    
    all_training_data=load_training_data()

    # Display menu and wait for user to start the game
    menu()

    # Let user play the game for 5 times to collect training data
    for i in range(5):
        print(f"Playing game {i+1}/5")
        
        training_data_iter=collect_training_data() 
        all_training_data.extend(training_data_iter) 


    # Train the model with collected data
    if all_training_data:
        print("Training the model")
        train_model(model, all_training_data, epochs=10)  # Reduce epochs for initial testing

        # Automatic Play with Neural Network
        print("Starting automatic play")
        automatic_play()
        print("Saving training data")
        save_training_data(all_training_data, f'{pathlib.Path(__file__).parent.resolve()}/training_data.pkl')


if __name__ == "__main__":
    # all_training_data=load_training_data()
    # if all_training_data:
    #     print("Training the model")
    #     train_model(model, all_training_data, epochs=10)  # Reduce epochs for initial testing

    #     # Automatic Play with Neural Network
    #     print("Starting automatic play")
    #     automatic_play()
    main()

