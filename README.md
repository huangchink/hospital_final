<div align="center">
  <a href="https://xmake.io">
    <img width="160" height="160" src="gazefollower/res/image/gazefollower.png">
  </a>

  <h1>GazeFollower</h1>

  <b>An open-source gaze tracking system for web cameras</b><br/>
  <i>Simple, Fast, Pythonic, Accurate</i><br/>
</div>

## Introduction

**GazeFollower** is a powerful and easy-to-use gaze tracking system designed specifically for use with web cameras. It offers an intuitive Python API, allowing developers and researchers to integrate gaze tracking into their projects with minimal setup. GazeFollower provides tools for real-time gaze tracking, calibration, and data recording, making it ideal for applications in psychology, usability testing, and more.

## Features

- **Accurate Tracking**: Achieves high accuracy and precision with built-in calibration methods.
- **Pythonic API**: Easy-to-use, with functions for common tasks like calibration and data saving.
- **Lightweight & Fast**: Optimized for real-time performance, ensuring smooth operation on most systems.
- **Experiment Ready**: Includes methods for triggering and saving data, ideal for experiment-based applications.

## Installation

You can install GazeFollower via pip or by cloning the repository.

### Installing with pip

```bash
python -m pip install gazefollower
```

### Git clone from Github

```bash
git clone https://github.com/GanchengZhu/GazeFollower
cd GazeFollower
python setup.py install
```


## Quick Start

Here's a basic example of how to use GazeFollower:

```python
#!/usr/bin/bash
# _*_ coding: utf-8 _*_

from gazefollower import GazeFollower
gaze_follower = GazeFollower()

gaze_follower.preview()
gaze_follower.calibrate()

gaze_follower.start_sampling()
# your experiment code
gaze_follower.send_trigger(10)
# your experiment code
gaze_follower.stop_sampling()
gaze_follower.save_data("demo.csv")
gaze_follower.release()
```

## Note

This depository only contains a model train on 7 million images. To gain access to the base model trained on 32 million images, please send an email to zhiguo@zju.edu.cn. Upon successful processing of your request, you will receive an email containing the model.

### Email Prompt

Here’s a template for your request email. Please keep the subject line unchanged:
```
Subject: Request for Access to the Base Model Trained on 32 Million Images

Dear Prof. Zhiguo,

I hope this message finds you well.

My name is [Your Name], and I am a [student/researcher] at [Your Affiliation]. I am writing to request access to the base model trained on 32 million images.

I assure you that I will use this model solely for academic and research purposes and will not utilize it for commercial activities or share it with others.

Thank you for considering my request. I look forward to receiving access to the model.

Best regards,
[Your Name]
```
