# Custom-OpenAI-Chatbot

## Project Details
In this project created a custom OpenAI chatbot that incorporates the articel 
[2023 Turkey-Syria earthquakes](https://en.wikipedia.org/wiki/2023_Turkey-Syria_earthquakes) from Wikipedia.

The notebook shows the difference of the model's answers depending on whether a custom prompt or a basic prompt is used. The functions for data cleaning, creating the embeddings and constructing the prompt for a specific question can be found in `utils.py`.

At the end of the notebook a simple chat bot is provided that allows to send repeatedly questions. The bot can be left by pressing the enter key.

## Getting Started

### Project File Structure
The project is structured as follows:

📦project<br>
 ┣ 📂data  **`(contains all source and result data files)`** <br>
 ┃ ┣ 📂source<br>
 ┃ ┃ ┗ ...<br>
 ┃ ┗ 📂results<br>
 ┃   ┗ ... <br>
 ┣ .gitignore <br>
 ┣ utils.py **`(several utility functions used in the notebook)`** <br>
 ┣ project.ipynb **`(Jupyter notebook with the chat bot a the end)`** <br>
 ┣ requirements.txt <br>
 ┗ README.md <br>

### Installation, Dependencies and Starting the Notebook

The code of this project was tested on Linux (Ubuntu 20.04). To get the code running on your local system, follow these steps which are base on Anaconda and pip:

1.  `conda create --name chatbot python=3.9 -c conda-forge`
2.  `conda activate chatbot`
3.  Create a directory where you want to save this project
4.  `git clone https://github.com/rp-dippold/Custom-OpenAI-Chatbot.git`
5.  `cd Custom-OpenAI-Chatbot`
6.  `pip install -r requirements.txt`
7.  `python -m ipykernel install --user --name chatbot --display-name "chatbot"`
8.  `jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10`
