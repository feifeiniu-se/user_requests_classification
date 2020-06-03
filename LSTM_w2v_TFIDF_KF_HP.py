import numpy
from gensim.models import Word2Vec
from keras import Input, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Embedding, LSTM, Dense, concatenate
from keras.optimizers import Adam
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn import preprocessing
from LSTM_5Fold.Read5F import readALL5
from LSTM_5Fold.Write_5Fold import Write5Fold2
from Write import Write

def LSTM_TFIDF_kw_HP_classify(TEXT_DATA, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, BATCH_SIZE, EPOCH):
    x_train, y_train, HPs_train, keyword_train, tfidf_train, bert_train = TEXT_DATA[0]
    x_test, y_test, HPs_test, keyword_test, tfidf_test, bert_test = TEXT_DATA[1]

    W2V_MODEL = Word2Vec(x_train, sg=1, size=EMBEDDING_DIM, window=7, min_count=0, negative=5, sample=0.00025, hs=1)
    tfidf_train = numpy.array(tfidf_train)
    tfidf_test = numpy.array(tfidf_test)
    k_train = numpy.array(keyword_train)
    k_test = numpy.array(keyword_test)
    HP_train = numpy.array(HPs_train)
    HP_test = numpy.array(HPs_test)

    # print("Tokenizer----------")
    tokenizer = Tokenizer()
    requires = x_train + x_test
    tokenizer.fit_on_texts(requires)
    word_index = tokenizer.word_index

    # print("Pad_sequence----------")
    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
    y_train = np_utils.to_categorical(y_train, num_classes=7)
    y_test = np_utils.to_categorical(y_test, num_classes=7)

    normalizer = preprocessing.Normalizer(norm='l1').fit(tfidf_train)
    tfidf_train = normalizer.transform(tfidf_train)
    tfidf_test = normalizer.transform(tfidf_test)
    #
    # normalizer = preprocessing.Normalizer().fit(keyword_train)
    # keyword_train = normalizer.transform(keyword_train)
    # keyword_test = normalizer.transform(keyword_test)

    # min_max_scaler = preprocessing.MinMaxScaler()
    # k_train = min_max_scaler.fit_transform(k_train)
    # k_test = min_max_scaler.transform(k_test)  # 归一化处理

    # print('Embedding----------')
    nb_words = len(word_index)
    embedding_matrix = numpy.zeros((nb_words + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        try:
            embedding_vector = W2V_MODEL.wv[word]
        except KeyError:
            continue
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # embedding层
    embedding_layer = Embedding(nb_words + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False
                                )


    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), name='inputs')#main input
    HP_input = Input(shape=(len(HP_train[0]),), name='HP_input')#auxiliary input
    keyword_input = Input(shape=(7,), name='keyword_input')
    tfidf_input = Input(shape=(len(tfidf_train[0]),), name='tfidf_input')

    sentence_input = embedding_layer(inputs)
    lstm_out = LSTM(64, name="lstm_out")(sentence_input)
    x = concatenate([lstm_out, tfidf_input, keyword_input, HP_input])
    dense1 = Dense(128, activation='relu')(x)
    predictions = Dense(7, activation='softmax')(dense1)

    adam = Adam(lr=0.01)
    model = Model(inputs=[inputs, tfidf_input, keyword_input, HP_input], outputs=predictions)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    early_stopping = EarlyStopping(monitor='acc', patience=3)
    reduceLROnPlateau = ReduceLROnPlateau(monitor='acc', factor=0.1, patience=3, mode='auto',
                                          min_delta=0.0001, cooldown=0, min_lr=0)
    model.fit([x_train, tfidf_train, k_train, HP_train], y_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, callbacks=[early_stopping, reduceLROnPlateau])
    score = model.evaluate([x_test, tfidf_test, k_test, HP_test], y_test)
    y_predict = model.predict([x_test, tfidf_test, k_test, HP_test])

    y_true = Write.one_hot_reverse(y_test)
    y_pred = Write.one_hot_reverse(y_predict)

    print("score:", score)

    return score, y_true, y_pred

def LSTM_TFIDF_kw_HP_classify_5Fold(NAME, DATA, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, BATCH_SIZE, EPOCH):
    i = 0
    true = []
    pred = []
    scores = []
    total_score = 0
    while (i < 5):
        data = DATA[i]
        score, y_true, y_pred = LSTM_TFIDF_kw_HP_classify(TEXT_DATA=data,
                                                          MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                                                          EMBEDDING_DIM=EMBEDDING_DIM,
                                                          BATCH_SIZE=BATCH_SIZE,
                                                          EPOCH=EPOCH)
        true = true + y_true
        pred = pred + y_pred
        scores.append(score)
        total_score = total_score + score[1]
        i += 1
    print("scores:", scores)
    avg_score = total_score/5
    each_acc, each_precision, each_f = Write.P_R_F(true, pred)
    Write5Fold2([NAME, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, EPOCH, BATCH_SIZE], avg_score, each_acc, each_precision, each_f, "../Result/LSTM/LSTM_w2v_KTH_5F.txt")
    return avg_score, each_acc, each_precision, each_f

name = "keepass"
MAX_SEQUENCE_LENGTH = 35
EMBEDDING_DIM = 64
epoch = 200
batch_size = 64
data = readALL5(name)
i = 0
while i < 3:
    i += 1
    a, r, p, f = LSTM_TFIDF_kw_HP_classify_5Fold(NAME=name,
                             DATA=data,
                             MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
                             EMBEDDING_DIM=EMBEDDING_DIM,
                             BATCH_SIZE=batch_size,
                             EPOCH=epoch)
    print("accuracy: ", a)
