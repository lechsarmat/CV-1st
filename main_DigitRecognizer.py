import pandas as pd
import func_DigitRecognizer as dr

train_df = pd.read_csv( "data/short_mnist_train_99.csv" )

P_t, p_t = dr.LearningAlg( train_df, iter=50, eps=7e-2, eta=1e-1 )
print(P_t, p_t, sep = '\n')