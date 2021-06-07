# Learning to Drive by Crashing
Robotics course project @ USI 20/21 - Using a CNN model to replace the proximity sensors for controlling Thymio movement in Gazebo

Authors: Anxhela Vëndresha, Arshjot Singh Khehra, and Srihari Sridharan

Some example videos have been provided in the `Videos` directory.

## Steps for running:

Note for running `cafe`, `warehouse`, `small_house`, and `random_items` worlds, you will need to first follow the models folders from the following links to `~/.gazebo/models`:

`cafe`: https://github.com/osrf/gazebo_models/tree/master/cafe
`small_house`: https://github.com/aws-robotics/aws-robomaker-small-house-world/tree/ros1/models
`warehouse`: https://github.com/aws-robotics/aws-robomaker-small-warehouse-world/tree/master/models
`random_items`: https://github.com/aws-robotics/aws-robomaker-hospital-world/tree/master/models


1. Place the `learning_to_drive' folder in `~/catkin_ws/src/`

2. Run:
    ```bash
    cd ~/catkin_ws && catkin build && re.
    ```
    
3. Run the below command for starting data collection in random world:
    ```bash
    roslaunch learning_to_drive generate_training_data.launch
    ```
    for warehouse:
    
    ```bash
    roslaunch learning_to_drive generate_training_data_specify_world.launch world:=warehouse
    ```
    
    for cafe:
    
    ```bash
    roslaunch learning_to_drive generate_training_data_cafe.launch
    ```
    
    for random items:
    
    ```bash
    roslaunch learning_to_drive generate_training_data_specify_world.launch world:=random_items
    ```

4. Run the below command for testing the CNN model in random world:

    ```bash
    roslaunch learning_to_drive test_cnn_model.launch
    ```
    
    for warehouse:
    ```bash
    roslaunch learning_to_drive test_cnn_model_other_envs.launch world:=warehouse
    ```
    
    for cafe:
    ```bash
    roslaunch learning_to_drive test_cnn_model_cafe.launch
    ```
    
    for small house:
    ```bash
    roslaunch learning_to_drive test_cnn_model_other_envs.launch world:=small_house
    ```

    for random items:
    ```bash
    roslaunch learning_to_drive test_cnn_model_other_envs.launch world:=random_items
    ```

5. For training the model, first run:
    ```bash
    python 'Data Preparation.py'
    ```
   and then run:
    ```bash
    python train.py
    ```