
# データ作成用クラス
import race_dnn as dn
import race_cnn as cn

para = 0

for jyuni in range(1, 7):
    print('{:0=2}'.format(para) + '-' + str(jyuni))

    dn.start_learning('train', 'test', 'dnn-model', 'dnn-test-csv', 'dnn-waku_rank', '{:0=2}'.format(para), str(jyuni), "./", True)
    cn.start_learning('train', 'test', 'cnn-model', 'cnn-test-csv', 'cnn_waku_rank', '{:0=2}'.format(para), str(jyuni), "./", True)


