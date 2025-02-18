This is going to be a project for CS445 Machine Learning, in this proejct we aim to train, test, and validate that a reinforcement model can be made to beat the top-down twin stick shooter game GeoMania (https://nik-fot.itch.io/geomania). 


Goals:

- Create a model that can interact with the game itself using OpenCV, Gym, Python, and DQN (Deep Q-learning)
- learn more about vision processing and Deep Learning
- Make a model that can play the game and improve based on reinforcement learning techniques
- Beat Geomanias fixed 20 wave limit
- Create a "Black-Box" model that has no access to back end code and uses image processing for decisions rater than logs and system information

Sprint overviews

  Sprint 1 (2/16-3/2)
  - Playtesting Geomania to understand core functionality
  - Understanding limitations with browser game vs using Unity
  - Download and install packages expected for project
  - Being preprocessing
      - Using OpenCV to take screen captures, find fixed capture resolutions, grayscale, etc.
  

  Sprint 2 (3/2-3/16)
  - Begin writing actions, rewards, and interactions within the game
  - Basic actions such as movement (wasd), aim (mouse input), and shooting
  - Set up basic reward functionality (points, time alive, enenmies killed, dying, etc)
  - Game inputs such as store functions (buying upgrades, moving onto next wave, restarting the game, etc.)

  Sprint 3 (3/16-3/30)
  - Being training model, see limitations with game bugs, system limitations, game speed, image processing, etc.
  - Recognize possible changes in reward system, see how the model handles mouse movement
  - Make a deicision on whether or not the need for unity backend code is essential for training/efficiency (for example, speeding up game speed for model training)

  Sprint 4 (3/30-4/13)
  - If shift to unity, reset up environment and possibly coding backend scripts to track enemy locations, player locations, etc.
  - If no shift to unity, find improvement to training, making sure actions are correct, continue training

  Sprint 5 (4/13-4/27)
  - Unsure: improvements as need be

  Sprint 6(4/27-5/11)
  - Unsure: improvements as need be

