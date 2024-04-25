# Robo Angle Seeker

## Overview
This repository accompanies the research paper "Precision Angle Seeking in Robots: A Reinforcement Learning Approach in Simulation and Reality." Our study investigates the application of Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL) algorithms for precision angle seeking in robotic control, highlighting the transition from simulations to real-world applications. The "Angular Positioning Seeker (APS)" environment, built on stm32f103 and Raspberry Pi 4B+ platforms, serves as a novel benchmark for assessing RL algorithms under conditions that closely emulate real-world scenarios.

## Features
- Advanced RL and DRL algorithms for precision control of robotic angles.
- The APS environment, a custom simulation tailored for angle seeking rather than balance.
- Integration with Raspberry Pi and STM32F103ZET6 microcontroller for real-world application.
- Comprehensive experimental setups for both simulated and physical environments detailed in the paper.

## Getting Started
### Prerequisites
- Python 3.x
- PyTorch (for specific algorithm implementations)
- Hardware: Raspberry Pi 4B+, STM32F103ZET6 microcontroller, L298N motor driver, DC3V-6V geared DC motors (TT motors), and additional components as listed in the paper's appendix.

### Installation
Clone the repository:
   ```
   git clone https://github.com/Ethan-Chen-plus/RoboAngleSeeker.git
   ```


## Usage

Usage examples include detailed command lines for running different DQN models and configurations as presented in the research, facilitating replication of the experiments and further exploration of the algorithms' behaviors in the APS environment.

1. **Default DQN Training**
   ```bash
   python main.py --model DQN
   ```

2. **Double DQN with Custom Parameters**
   ```bash
   python main.py --model DoubleDQN --lr 0.005 --num_episodes 300 --epsilon 0.05
   ```

3. **Dueling DQN with Increased Batch Size and Buffer Size**
   ```bash
   python main.py --model DuelingDQN --batch_size 128 --buffer_size 10000
   ```

## Experiment and Results
The benchmark demonstrates the effectiveness and challenges of RL and DRL algorithms in physical settings, as discussed extensively in our paper. Results provide insights into the nuances of algorithm performance across simulated and real environments.


## Citation
Please cite our work if it assists in your research:
```
@inproceedings{chen2024precision,
  title={Precision Angle Seeking in Robots: A Reinforcement Learning Approach in Simulation and Reality},
  author={Kewei Chen, Shai Li, Mingsheng Shang},
  year={2024},
  booktitle={Proceedings}
}
```

## License
This project is licensed under the MIT License - see the `LICENSE` file for more details.

## Acknowledgments
- Acknowledgments to funding bodies, research groups, and any other support should be listed here.
