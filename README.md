# acds-day-after-tomorrow-hugo

## Contents

* [Requirements](#requirements)
* [About the project](#about-the-project)
* [Project structure](#project-structure)
* [Dataset](#dataset)
* [LICENSE](#license)
* [About us](#about-us)

## Requirements

* Run on Google Colab(high memory, high RAM)
* Can also be run locally, should include environment.yml

## About the project

Hurricanes can cause upwards of 1,000 deaths and $50 billion in damages in a single event, and have been responsible for well over 160,000 deaths globally in recent history. During a tropical cyclone, humanitarian response efforts hinge on accurate risk approximation models that can help predict optimal emergency strategic decisions.


## Project structure

```
 ├── .github
 │   ├── workflows
 │   │   ├── flake8.yml - Checking Python code for compliance with the PEP8 specification
 │   │   ├── tests.yml - pytest
 │   │   ├── sphinx.yml - try to generate the documentation
 ├── README.md - Introduction of the project
 ├── HurricaneForecast - Packages  
 ├── docs - the documentation
 ├── test
 │   ├── test_model.py - pytest
 ├── environment.yml
 ├── requirements.txt
 ├── setup.py
 ├── setup.py
 ├── HurricaneForecast_Tool.ipynb - finial version of work not analysis
 ├── Analysis_notebook.ipynb
 Action
 └── LICENSE
 
```

## Dataset
* NASA Satellite images of tropical storms
* 494 storms around the Atlantic and East
Pacific Oceans (precise locations
undisclosed)
* Each with varied number of time samples
(4 - 648, avg 142)
* Labelled by id, ocean (1 or 2) and wind
speed

|image_id|storm_id|relative_time|ocean|wind_speed|
|  ----  | ----  | ----  | ----  | ----  |
| 0  | abs_000 | abs | o | 2 | 43 |
| 1  | abs_000 | abs | 1800 | 2 | 44|
| 2  | abs_000 | abs | 5400 | 2 | 45|
| 3  | abs_000 | abs | 17999 | 2 | 52|
| 4  | abs_000 | abs | 19799 | 2 | 53|



## LICENSE

Distributed under the MIT License. See LICENSE for more information.

## About us

Team name: Hugo

Team members: Wan Ian I, Castaneda Angela, Wang Ziyou, Kanimova Ellya, Pu Enze, Zhang Peng, Cai Wan, You Zhexin, Kong Xinyan

Project Repository: https://github.com/ese-msc-2021/acds-day-after-tomorrow-hugo/tree/main
