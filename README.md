# ODS Classification
Modelo para clasificacion de ODS

## Data and model
### Enlace:
- https://clasificaods.espol.edu.ec/
- https://i-research.espol.edu.ec/odss

### Dataset: 
https://zenodo.org/records/11441197

### Entrenamiento bert: 
https://medium.com/@shaikhrayyan123/a-comprehensive-guide-to-understanding-bert-from-beginners-to-advanced-2379699e2b51

LabelEncoder = para llevar a id los label de salida

### Clasificador:
https://huggingface.co/docs/transformers/model_doc/bert

### Traductor:
https://huggingface.co/docs/transformers/model_doc/marian = MarianMTModel
https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-es
https://huggingface.co/frluquba/traductor_es_en

### Data augmentation:
    https://github.com/makcedward/nlpaug

    - NLTK, nlpaug y nagisa
    pip install numpy requests nlpaug nagisa

    pip install nltk

    - Descargar semi manualmente
    https://www.nltk.org/data.html
    >>> import nltk
    >>> nltk.download()



- Nlpaug: 
Purpose: Textual data augmentation for NLP models. 
Functionality: Introduces variations in text data (e.g., synonym replacement, word insertions) to increase the size and robustness of training datasets. 
Use cases: Improving the performance of deep learning models for tasks like text classification, named entity recognition, and question answering. 
Example: Augmenting sentences by replacing words with synonyms to create more diverse training examples. 

- pandas DataFrame.sample():
Purpose: Randomly selecting rows from a DataFrame. 
Functionality: Returns a new DataFrame containing a random subset of the original DataFrame's rows. 
Use cases: Data exploration, creating smaller representative datasets for testing, or creating batches of data for model training. 
Example: Selecting 100 random rows from a DataFrame with 1000 rows to create a smaller dataset for experimentation. 

- Key Differences: 
Data Type: Nlpaug works with text data, while DataFrame.sample() works with tabular data represented in a pandas DataFrame. 
Purpose: Nlpaug is for data augmentation to improve model training, while DataFrame.sample() is for data sampling for exploration and analysis. 
Output: Nlpaug generates new text data, while DataFrame.sample() returns a subset of the original DataFrame. 

## Structure
- models
- results
- plots
- data
- diferent_models
- imgs
- docs
- kaggle
- .gitignore
- README.md
- requirements.txt
- roberta.ipynb
- bert.ipynb
- trainig.log
- 1_process_data.ipynb
- 2_text_augmented.ipynb
- 3_data_analytics.ipynb
- 4_all_models.ipynb
- 5_use_model.ipynb
