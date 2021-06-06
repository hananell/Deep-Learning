Welcome to our bilstm!

the program bilstmTrain run with 3 arguments:
1:  'a', 'b', 'c', or 'd' - for representation kind.
2:  'ner' or 'pos' - for file kind to train on.
3:  desired name of trained model file produced.

the train/dev/test data must be in directory 'pos' or 'ner', and that directory must be in the same place as the scripts. for example the script will search for 'ner/train'.
utils.py has helper functions for bilstmTrain.

the run produces 2 graphs - for loss and for accuracy at each epoch, and trained model.


the program bilstmPredict run with 3 arguments:
1:  'a', 'b', 'c', or 'd' - for representation kind.
2:  name of model file to load and use.
3:  'ner' or 'pos' - for file kind to train on.

the run produces test4.{inputFile} file which is the test file with predicted labels.