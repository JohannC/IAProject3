    # -*- coding: utf-8 -*-
    """
    Created on Mon Dec  4 17:10:52 2017
    
    @author: Lorenzo
    """
    
    import pandas as pd
    import re
    import numpy as np
    import gc
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import GridSearchCV
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import opinion_lexicon
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    
    def review_to_words( review_text, set ):
        # Function to convert a raw review to a string of words
       
        letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
        words = letters_only.lower().split()                                            
        meaningful_words = [w for w in words if w in set]
        return( " ".join( meaningful_words ))  
    
    
    train1 = pd.read_csv("fulldata_neg.txt", header=0, delimiter="\n")
    train2 = pd.read_csv("fulldata_pos.txt", header=0, delimiter="\n")
    neg = [0] * 12500
    pos = [1] * 12500
    dat = pd.DataFrame({'feel': neg})
    dat2 = pd.DataFrame({'feel': pos})
    train1 = train1.join(dat)
    train2 = train2.join(dat2)
    frames = [train1, train2]
    training_set = pd.concat(frames,ignore_index=True)
    
    
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
    
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0, shuffle = True)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    
    
    
    test1 = pd.read_csv("fulldata2_neg.txt", header=0, delimiter="\n")
    test2 = pd.read_csv("fulldata2_pos.txt", header=0, delimiter="\n")
    neg = [0] * 12500
    pos = [1] * 12500
    da = pd.DataFrame({'feel': neg})
    da2 = pd.DataFrame({'feel': pos})
    test1 = test1.join(da)
    test2 = test2.join(da2)
    frames_test = [test1, test2]
    test_set = pd.concat(frames_test,ignore_index=True)
  
    num_reviews = test_set["Review"].size
    clean_test_reviews = []
    
    
    for i in range( 0, num_reviews ):
        clean_test_reviews.append( review_to_words( test_set["Review"][i], opinions ))
        
    
    test_data_features = vectorizer.fit_transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    
    resp_y = test_set.iloc[:, 1].values
    test_X = test_data_features
    
    sc = StandardScaler()
    test_X = sc.fit_transform(test_X)
    
    
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.optimizers import SGD
    from keras.constraints import max_norm
    
    sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
    
    def build_classifier(optimizer):
        classifier = Sequential()
        classifier.add(Dropout(0.2, input_shape=(500,)))
        classifier.add(Dense(150, kernel_constraint=max_norm(3.)))
        classifier.add(Dropout(0.3))
        classifier.add(Dense(150, kernel_constraint=max_norm(3.)))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(150, kernel_constraint=max_norm(3.)))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        return classifier
    classifier = KerasClassifier(build_fn = build_classifier)
    parameters = {'batch_size':[50,100,250],
                  'nb_epoch':[50,100,150],
                  'optimizer':['adam','rmsprop',sgd]}
    grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
    
    grid_search = grid_search.fit(test_X, resp_y)
    best_parameters = grid_search.best_params_ 
    best_accuracy = grid_search.best_score_

  
    

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
 
    
    
    
    y_pred_r = classifier.predict(test_X)
    y_pred_r = (y_pred_r > 0.5)
    
    from sklearn.metrics import confusion_matrix
    cmtest = confusion_matrix(resp_y, y_pred_r)



    

