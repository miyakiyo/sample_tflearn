# レースの解析を始める



def start_learning(trainFile, testFile, modelOutFile, csvOutFile, learnType, para, jyuni, dataPath = "", modelFlag = False):
    # TensorFlowライブラリ
    import tensorflow as tf
    # tflearnライブラリ
    import tflearn
    # 配列復元
    import pickle

    import csv
    import numpy as np
    from datetime import datetime

    # 層の作成、学習に必要なライブラリの読み込み
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.normalization import local_response_normalization
    from tflearn.layers.estimator import regression

    # 学習とテストのデータ読み込み
    with open(dataPath + 'data/X-' + trainFile + '-' + para + 'X.dump', 'rb') as fb:
        trainX = pickle.load(fb)

    with open(dataPath + 'data/Y-' + trainFile + '-' + para + '-' + jyuni + 'Y.dump', 'rb') as fb:
        trainY = pickle.load(fb)

    with open(dataPath + 'data/X-' + testFile + '-' + para + 'X.dump', 'rb') as fb:
        testX = pickle.load(fb)

    with open(dataPath + 'data/Y-' + testFile + '-' + para + '-' + jyuni + 'Y.dump', 'rb') as fb:
        testY = pickle.load(fb)

    print ('-->train')
    print (len(trainX))
    print (len(trainX[0]))
    print (trainX[0])
    print (len(trainY))
    print (trainY[0])
    print ('-->test')
    # print (testY)
    print(testY.argmax(axis=1))

    trainX = trainX.reshape([-1, 6, 9, 1])
    testX = testX.reshape([-1, 6, 9, 1])

    print ('-->train reshape')
    print (len(trainX))
    print (len(trainX[0]))
    print (trainX[0])

    # テスト結果保存
    testData = []

    with open(dataPath + 'csv/accuracy.txt', mode='a') as f:
        f.write(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + ' -> START \n')

    ### 学習とテスト
    with tf.Graph().as_default():

        # TensorFlowの初期化
        # tf.reset_default_graph()

        ## 入力層の作成
        net = tflearn.input_data(shape=[None, 6, 9, 1])

        ## 中間層の作成
        # 畳み込み層の作成
        net = conv_2d(net, 32, 5, activation='relu')
        # プーリング層の作成
        net = max_pool_2d(net, 2)
        # 畳み込み層の作成
        net = conv_2d(net, 64, 5, activation='relu')
        # プーリング層の作成
        net = max_pool_2d(net, 2)
        # 全結合層の作成
        net = fully_connected(net, 128, activation='relu')
        net = dropout(net, 0.9)

        ## 出力層の作成
        net = tflearn.fully_connected(net, 6, activation='softmax')
        net = tflearn.regression(net, optimizer='sgd', learning_rate=0.5, loss='categorical_crossentropy')

        # 学習の実行
        model = tflearn.DNN(net)
        model.fit(trainX, trainY, n_epoch=30, batch_size=100, validation_set=0.1, show_metric=True)

        # model(学習データ)の保存
        if (modelFlag):
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.save(sess, dataPath + 'model/cnn/' + modelOutFile + '-' + para + '-' + jyuni + '.ckpt')

        print ('-->test')
        # テスト開始
        for test in testX:
            testData.append(model.predict([test]))
            
        # 結果をCSVに保存
        with open(dataPath + 'csv/ai-yoso/cnn/' + csvOutFile + '-' + para + '-' + jyuni + '.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
            for line in testData:
                for col in line:
                    writer.writerow(col) # 2次元配列も書き込める
    
        # 予測
        pred = np.array(model.predict(testX)).argmax(axis=1)
        print(pred)
        
        label = testY.argmax(axis=1)
        print(label)
        
        accuracy = np.mean(pred == label, axis=0)
        print(accuracy)

        with open(dataPath + 'csv/accuracy.txt', mode='a') as f:
            f.write(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + ' , ' + learnType  + '_' +  para + '_' + jyuni + ' , ' + str(accuracy) + '\n')




