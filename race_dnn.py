# レースの解析を始める

# , para, jyuni は　ファイルの命名だけに使用
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

    # テスト結果保存
    testData = []

    with open(dataPath + 'csv/accuracy.txt', mode='a') as f:
        f.write(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + ' -> START \n')

    ### 学習とテスト
    with tf.Graph().as_default():
        # TensorFlowの初期化
        net = tflearn.input_data(shape=[None, 54])

        # activation = linear relu
        net = tflearn.fully_connected(net, 256, activation='relu')

        net = tflearn.fully_connected(net, 256, activation='relu')

        # net = tflearn.fully_connected(net, 128, activation='relu')

        # dropout 全体の何割を残すか
        net = tflearn.dropout(net, 0.9)

        # net 答えの数 activation = sigmoid softmax
        net = tflearn.fully_connected(net, 6, activation='softmax')
        net = tflearn.regression(net, optimizer='sgd', learning_rate=2., loss='mean_square')

        # 学習開始
        # n_epoch ≒ 学習回収
        # validation_set テストデータセットの割合
        model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='./board')
        # model.fit(trainX, trainY, n_epoch=20, validation_set=0.1, snapshot_epoch=False)
        model.fit(trainX, trainY, n_epoch=30, validation_set=0.1, 
                snapshot_epoch=False, show_metric=True, run_id='dense_model')

        # model(学習データ)の保存
        if (modelFlag):
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.save(sess, dataPath + 'model/dnn/' + modelOutFile + '-' + para + '-' + jyuni + '.ckpt')

        print ('-->test')
        # テスト開始
        for test in testX:
            testData.append(model.predict([test]))
            
        # 結果をCSVに保存
        with open(dataPath + 'csv/ai-yoso/dnn/' + csvOutFile + para + jyuni + '.csv', 'w') as f:
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



