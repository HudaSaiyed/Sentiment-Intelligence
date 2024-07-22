# Sentiment-Intelligence
### Summary of "Sentiment Intelligence: Analyzing Textual Data"

The "Sentiment Intelligence: Analyzing Textual Data" project is designed to process and analyze text data for sentiment and emotion detection. The project is divided into two main parts, each leveraging various natural language processing (NLP) and machine learning techniques.

The first part of the project focuses on emotion detection from textual data. A predefined emotions file, categorizing various emotions into base groups using a dictionary, serves as the foundation. The system reads a text file, `read.txt`, processes its content by removing punctuation and stop words, and tokenizes the text. The processed text is then matched against the emotion list to count the occurrence of each base emotion. Visualization tools like Matplotlib and Seaborn are used to create dynamic bar graphs representing the frequency of emotions, offering a clear visual insight into the emotional tone of the text.

The second part of the project involves sentiment analysis on a dataset containing text reviews. Each review is labeled as positive, negative, or neutral. The data undergoes cleaning and preprocessing, including stemming and lemmatization using NLTK. After splitting the data into training and test sets, machine learning models such as Support Vector Machine (SVM) and Random Forest are employed to classify the sentiments. The accuracy and performance of these models are evaluated to ensure reliable sentiment predictions.

A web application was developed using Streamlit to provide an interactive interface for both parts of the project. The application allows users to input text for emotion and sentiment analysis and view the results through dynamic visualizations. Streamlit facilitates a seamless deployment process without the need for a separate backend service.

Overall, the project demonstrates an effective approach to understanding and visualizing the emotional and sentimental content of textual data, with practical applications in various fields such as customer feedback analysis, social media monitoring, and more.
