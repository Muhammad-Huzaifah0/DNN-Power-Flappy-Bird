# Automated Flappy Bird Game with Deep Learning

This project implements a Flappy Bird game using Pygame and integrates a neural network model using TensorFlow to automate the gameplay. The model learns to play the game by collecting training data and then using this data to make decisions during the game.

## Features

- Basic Flappy Bird game implemented using Pygame.
- Integration of a neural network model using TensorFlow.
- Automatic gameplay using the trained model.
- Collection of training data from user gameplay.
- Menu system for starting the game.

## Installation

To run this project, you need to have Python installed along with the necessary libraries. You can install the required libraries using the following commands:

```bash
# Clone the repository
git clone https://github.com/YourUsername/Automated-Flappy-Bird-Game.git
cd Automated-Flappy-Bird-Game

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required libraries
pip install -r requirements.txt
```
## Usage
To start the game and begin training the model, run the following command:
```bash
python Flappy_Bird.py
```
## Controls
- Spacebar: Makes the bird jump.
- Mouse Click: Click the "Start" button on the menu to begin.

## Training Data
The game will allow you to play 5 rounds to collect training data. After that, the model will be trained using the collected data and will start playing automatically.

## Project Structure
- `Flappy_Bird.py`: Main game and training script.
- Assets/: Directory containing game assets like images and sounds.
- Raw/: Directory containing raw sound files.
- training_data.pkl: File where the collected training data is saved.

## Dependencies
```bash
Pygame
TensorFlow
NumPy
Pickle
```
## Acknowledgments
This project was inspired by the classic Flappy Bird game and integrates Deep learning to automate gameplay. Special thanks to the creators of Pygame and TensorFlow for their powerful libraries.