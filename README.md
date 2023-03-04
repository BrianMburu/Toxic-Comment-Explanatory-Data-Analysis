## Toxic Comment Text Analysis.

### Introduction

This Jupyter notebook provides a step-by-step guide to performing exploratory data analysis on the Toxic Comment dataset, and building a DistilBERT model for classifying toxic comments.

### Importing Libraries

This section imports all the necessary libraries for data cleaning, text processing, and data visualization. Some of the key libraries used in this notebook include:

- `re` for regular expressions
- `nltk` for natural language processing tasks such as tokenization and stemming
- `pandas` for data manipulation and analysis
- `seaborn` and matplotlib for data visualization
- `wordcloud` for generating word clouds
- `transformers` for loading and fine-tuning the DistilBERT model.

### Stemming / Lemmatizing and cleaning the text data

This section covers different techniques for preprocessing text data to make it suitable for analysis and modeling. Specifically, this section discusses three techniques for transforming the raw text into a more structured format:

- Porter stemmer: a popular stemming algorithm that reduces words to their base or root form. This algorithm is widely used in NLP tasks and has been shown to be effective in many applications.
- Snowball stemmer: a more aggressive stemming algorithm that is an extension of the Porter stemmer. This algorithm is designed to handle different languages and is more effective than the Porter stemmer in some cases.
- WordNetLemmatizer: a lemmatization algorithm that reduces words to their base form or lemma. This algorithm is more sophisticated than stemming and can handle different parts of speech.

### Exploratory Data Analysis (EDA) on the Toxic Comment Data

This section covers different exploratory data analysis techniques for gaining insights into the structure and patterns in the Toxic Comment dataset. Specifically, this section covers the following topics:

- Basic statistics: Compute some basic statistics about the data, such as the number of comments, the number of toxic comments, the average length of comments, and the distribution of toxic comments across categories (if available).
- Class distribution: Check the distribution of the target variable (toxicity) to ensure that there is a balance between the number of positive and negative samples. This is important for ensuring that the model does not overfit to one class.
- Correlation analysis: Check the correlation between different features or variables in the data to identify any patterns or relationships. For example, you can check if there is a correlation between comment length and toxicity.
- Word frequency: Check the frequency of the most common words in the corpus to get a sense of the vocabulary and what types of words are most commonly used. You can create a histogram of the word counts, or a word cloud to visualize the most common words.
- Relationship between target and most frequent words: Check if there are any words that appear more frequently in toxic comments than in non-toxic comments. You can create a bar chart or a word cloud to visualize the relationship.
- Word associations: Check if there are any words that are strongly associated with toxic comments. Using techniques such as association rule mining, Apriori algorithm, and co-occurrence analysis to identify these associations.
- Length distribution: Check the distribution of comment lengths to see if there are any patterns or differences between toxic and non-toxic comments. You can create histograms or box plots to visualize the length distribution.

### DistilledBert Model

This section covers the implementation of the DistilBERT model for classifying toxic comments. Specifically, this section covers the following topics:

- Load pretrained model state: Load the pretrained DistilBERT model state and configure the tokenizer to prepare the text data for input to the model.

- Making predictions: Use the loaded DistilBERT model to make predictions on the preprocessed text data. This involves converting the preprocessed text data to input features that can be passed to the model, using the tokenizer. The output of the model is a probability distribution over the different target labels (toxicity categories).

DistilBERT is a pre-trained transformer-based model that has shown promising results in various natural language processing (NLP) tasks. It is a smaller and faster version of BERT, with fewer parameters and faster inference time. DistilBERT uses a combination of a transformer encoder and a distillation process to produce a smaller and faster model that still maintains high performance. The use of pre-trained models like DistilBERT has been shown to improve the performance of NLP models on various text classification tasks. You can download the model I trained from this <a href="https://github.com/BrianMburu/Toxic-Comment-Classification-App.git">repo</a> in the models folder or train it yourself using instructions from this <a href="https://github.com/BrianMburu/Distiled-BERT-model-training-pytorch.git">repo</a>.

To load the pretrained model state, we use the DistilBERTClass class, which contains the DistilBERT model architecture and the necessary methods for preprocessing the input text data. We then load the model weights from the saved checkpoint file toxic_comment.pkl using PyTorch's load_state_dict function.

To make predictions on the preprocessed text data, we use the predict_text function. This function takes in the preprocessed text data, the loaded model, and the tokenizer, and produces the probability distribution over the target labels. The output of the function is a dictionary containing the predicted class labels and their corresponding probabilities.

### References:

- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.
- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Brew, J. (2020). Transformers: State-of-the-art Natural Language Processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (pp. 38-45).
