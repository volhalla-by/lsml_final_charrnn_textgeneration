# Text Generation with CharRNN

## Projecr Description
Reccurent Neural Network was trained to generate a text one character at the time. As training data, 99 most popular books from [Project Gutenberg](https://www.gutenberg.org/browse/scores/top) were used. You may find it in **data** folder under _archive_books.zip_ name. On the first step dictionaries for converting characters to integers and vice versa were created. LSTM expects one-hot encoded input, which means that each character is converted to an integer (via our created dictionary) and then converted to a vector, where a value 1 will be only on a corresponding position and the rest of the vector will be filled with zeros. To train the neural network, mini-batches were organised as follows: divide the entire input sequence by the desired number of subsequences (parameter _batch_size_), and send a sequence of length _seq_length_ to the input of the network.
![image](https://user-images.githubusercontent.com/9429871/167312573-d64d8691-940d-4422-96a4-9b771ca9e1c6.png)

### Model architecture

* Define an LSTM layer with dropout=drop_prob and batch_first=True 
* Define a Dropout layer with drop_prob.
* Define a Linear layer with in_features=n_hidden and out_features equals to number of characters.
* **Cross Entropy** as a loss function were users and **Adam** as optimizer

### Prediction
We predict the next character for the sequence of input symbols. We pass a character as input, and the network predicts the next character. Then we take that character, pass it as input, and get another predicted character, and so on. Our RNN's output comes from a fully connected layer and outputs the distribution of the next character scores. To actually get the next character, we use the softmax function, which gives us a probability distribution that we can then choose to predict the next character. Our predictions are based on a categorical distribution of probabilities for all possible characters. We can make the sampling process smarter by looking at only some of the most likely characters. This will prevent the network from giving us completely absurd characters, and will also allow some noise and randomness to be introduced into the selected text. This technique is called top K sampling.

### Results
The model were trained 3 epochs (5650 steps). After 5 hours of training loss function reached ```Loss: 1.5018, Val Loss: 1.3823```. 

## Application architecture
The app consists of the following services:
* **Redis** in-memory data stores of Celery data
* **Flask** is a web framework connected to Celery 
* **Celery** is a task queue implementation for Python web applications used to asynchronously execute work outside the HTTP request-response cycle
* **MLFlow** that record and query experiments with models
* Jupyter notebooks with model training

## Quick start
1. Building an images from docker-compose.yml
```docker-compose build --no-cache```
2. The docker-compose up command aggregates the output of each container 
```docker-compose up```
3. Application us running on localhost 5001 port (http://localhost:5001/). Just enter a starting sowrd or a phrase, click on submit button and wait until the resoinse.
<img width="1370" alt="image" src="https://user-images.githubusercontent.com/9429871/167312924-f64c6679-5853-45c3-b24b-74714a397303.png">
