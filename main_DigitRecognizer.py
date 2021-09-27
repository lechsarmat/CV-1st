import pandas as pd
import func_DigitRecognizer as dr

train_df = pd.read_csv( "data/short_mnist_train_99.csv" )
test_df = pd.read_csv( "data/short_mnist_test_99.csv" )

P_t, p_t = dr.LearningAlg( train_df, iter=50, eps=7e-2, eta=1e-1 )

train_res = dr.CheckAlg(train_df, P_t, p_t)
test_res = dr.CheckAlg(test_df, P_t, p_t)
print('Train res: '+str(train_res),'Test res: '+str(test_res), sep = '\n')