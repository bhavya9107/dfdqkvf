def process(x):
''' Enter argument as name of file (if file in root), or path of file'''
    import numpy
    from keras.datasets import imdb
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers.embeddings import Embedding
    from keras.preprocessing import sequence, text
    import pandas as pd
    from sklearn.cross_validation import train_test_split
    
    df=pd.read_csv(x)
    tk = text.Tokenizer(nb_words=200, lower=True)
    tk.fit_on_texts(df['Message'])

    df['Message'] = tk.texts_to_sequences(df['Message'])
    return df
df= process()