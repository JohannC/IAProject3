    # -*- coding: utf-8 -*-
    """
    Created on Mon Dec  4 17:10:52 2017
    
    @author: Lorenzo
    """
    
    import pandas as pd
    import re
    import numpy as np
    import keras
    from nltk.corpus import opinion_lexicon
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.optimizers import SGD
    from keras.constraints import max_norm
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix
    from sklearn.feature_extraction.text import CountVectorizer



    
    def review_to_words( review_text, set ):
        # Function to convert a raw review to a string of words
       
        letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
        words = letters_only.lower().split()                                            
        meaningful_words = [w for w in words if w in set]   
          
        return( " ".join( meaningful_words ))  
    
    
    #Reading the training data
    
    train_neg = pd.read_csv("fulldata_neg.txt", header=0, delimiter="\n")
    train_pos = pd.read_csv("fulldata_pos.txt", header=0, delimiter="\n")
    zeros = [0] * 12500
    ones = [1] * 12500
    data_zeros = pd.DataFrame({'feel': zeros})
    data_ones = pd.DataFrame({'feel': ones})
    train_neg = train_neg.join(data_zeros)
    train_pos = train_pos.join(data_ones)
    frames = [train_neg, train_pos]
    training_set = pd.concat(frames,ignore_index=True)
    

    # Cleaning the training data
    
    num_reviews = training_set["Review"].size
    clean_train_reviews = []
    
    opinions = set(opinion_lexicon.words())
    
    for i in range( 0, num_reviews ):
        clean_train_reviews.append( review_to_words( training_set["Review"][i], opinions ))
        
     
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 500) 
    
    
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()
    
    y = training_set.iloc[:, 1].values
    X = train_data_features
    
    # Split the data to do the training
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0, shuffle = True)
    
    # Scale the values to 0 -> 1
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    
    # Initialising the ANN
    classifier = Sequential()
    early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='auto')
    
    # Adding the input layer and two hidden layers with dropout
    
    classifier.add(Dropout(0.2, input_shape=(500,)))
    classifier.add(Dense(150, kernel_constraint=max_norm(3.)))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(150, kernel_constraint=max_norm(3.)))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(150, kernel_constraint=max_norm(3.)))
    
  
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
             
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 10, epochs = 10, callbacks=[early_stop])
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    # Showing the results
    cm = confusion_matrix(y_test, y_pred)
    
    # Reading the test data
    test_neg = pd.read_csv("fulldata2_neg.txt", header=0, delimiter="\n")
    test_pos = pd.read_csv("fulldata2_pos.txt", header=0, delimiter="\n")
    test_neg = test_neg.join(data_zeros)
    test_pos = test_pos.join(data_ones)
    frames_test = [test_neg, test_pos]
    test_set = pd.concat(frames_test,ignore_index=True)
    
    
    #Cleaning the test data
    num_reviews = test_set["Review"].size
    clean_test_reviews = []
    
    
    for i in range( 0, num_reviews ):
        clean_test_reviews.append( review_to_words( test_set["Review"][i], opinions ))
        
    
    test_data_features = vectorizer.fit_transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    
    resp_y = test_set.iloc[:, 1].values
    
    sc = StandardScaler()
    test_data_features = sc.fit_transform(test_data_features)
    
    #Predicting with test data
    
    y_pred_r = classifier.predict(test_data_features)
    y_pred_r = (y_pred_r > 0.5)
    
    # Showing the results
    
    cmtest = confusion_matrix(resp_y, y_pred_r)



    

