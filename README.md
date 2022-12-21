# Live privacy filter using Mediapipes

A live feed video filter in python using OpenCV and Mediatools with enfasis in privacy

## Getting Started

> There is not yet a release of this project, but, the files **sample_roulette.py** **face_thresholds.py** and **single_image_detection** are designed to showcase the diferent processes used to make this program function.

### Prerequisites

This files require Python3.10, OpenCv, Mediapipes, and other libraries that are already native to python

```python
import cv2
import numpy as np
import itertools
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

```

### Installing
---

**OpenCV**

To install OpenCv as a module with pip:

```
pip install opencv-python  
```
For instalation with Command Promt or Anaconda refer to [here.](https://www.javatpoint.com/how-to-install-opencv-in-python)

---
**Mediapipes**

The same process for Mediapipes

```
pip install mediapipes
```
For instalation with Command Promt or Jupyter Notebooks refer to [here.](https://omdena.com/blog/mediapipe-python-tutorial/)

## Showcase

Facial recognition and basic landmarks

![Three pictures of the same person, one regular, second with facial detection and third with complex facial detection and mesh][single_picture_showcase]
[single_picture_showcase] = //INSERT IMAGE HERE

The program goes trough the live video and obtains the position of the facial landmarks, then censors the selected landmarks and outputs the censored frames

## Deployment

>Not deployable yet

## Built With

* [OpenCV](https://opencv.org/) - Facial recognition and tracking for live video
* [Mediapipe](https://mediapipe.dev/) - Facial landmark recognition

**This code is heavily based in [this](https://bleedai.com/facial-landmark-detection-with-mediapipe-creating-animated-snapchat-filters/) Notebook by Bleed.AI**

## Contributing

>This is a personal project and there is no way to contribute just yet, altough feedback is apreciated

## Authors

* **Yair Salvador** - *Main developer* - [Y0z64](https://github.com/Y0z64)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [Bleed AI](https://bleedai.com/) - [Notebook](https://bleedai.com/facial-landmark-detection-with-mediapipe-creating-animated-snapchat-filters/)
