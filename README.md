## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Content](#content)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)


About the Project
---

In this project, we use and convolutional neural networks to classify traffic signs. We will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, it can be evaluated on any german traffic sign you find in the world wide web.

The steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

<!-- GETTING STARTED -->
## Getting Started

The software is written in Python 3.7 and tested on Linux. The usage of the Miniconda Python distribution is strongly recommended.

### Prerequisites

* Miniconda (https://docs.conda.io/en/latest/miniconda.html)

### Installation

1. Clone this repo
```sh
git clone https://github.com/sebkaster/CarND-Traffic-Sign-Classifier-Project.git
```

2. Create Anaconda environemnt
```sh
conda create -n traffic_sign_env anaconda python=3
```

3. Actiavate environment
```sh
conda activate traffic_sign_env
```

4. Install pip package manager
```sh
conda install pip
```

5. Install required python modules
```
python -m pip install -r requirements.txt
```

<!-- CONTENT -->
## Content

* new-test-images/: New traffic sign images found in the internet to evaluate the trained model.
* examples/: Images used in this Readme.md or in the writeup.
* _Traffic_Sign_Classifier.ipynb_: Jupyter notebook containing the whole code of the project.
* writeup.md: Documentation of the project. 

<!-- USAGE EXAMPLES -->
## Usage

In order to use this project start jupyter by typing `jupyter notebook` in your Anaconda environment. The notebook _Traffic_Sign_Classifier.ipynb_ shows the implementation of the preprocessing, training, and evaluation. Run the cells to see how it works.

For the training I used the dataset from here: [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the MIT License.

<!-- CONTACT -->
## Contact

Sebastian Kaster - sebastiankaster@googlemail.com

Project Link: [https://github.com/sebkaster/CarND-Traffic-Sign-Classifier-Project](https://github.com/sebkaster/CarND-Traffic-Sign-Classifier-Project



