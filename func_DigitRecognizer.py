import numpy as np
import pandas as pd

def LearningAlg(df, iter=10, eps=5e-2, eta=1e-1):
    """Implements EM-algorithm for clustering two classes of images.
      >>> np.max(LearningAlg(pd.read_csv( "data/mini_test.csv" ))[0], axis = 0)
      array([[0.95, 0.95, 0.95, 0.95],
             [0.95, 0.95, 0.95, 0.95],
             [0.95, 0.95, 0.95, 0.95],
             [0.95, 0.95, 0.95, 0.95]])
      >>> np.min(LearningAlg(pd.read_csv( "data/mini_test.csv" ))[0], axis = 0)
      array([[0.05, 0.05, 0.05, 0.05],
             [0.05, 0.05, 0.05, 0.05],
             [0.05, 0.05, 0.05, 0.05],
             [0.05, 0.05, 0.05, 0.05]])
      >>> LearningAlg(pd.read_csv( "data/mini_test.csv" ))[1]
      array([0.5, 0.5])
      >>> LearningAlg(0)
      Traceback (most recent call last):
        ...
      TypeError: df must be pandas dataframe
      >>> LearningAlg(pd.read_csv( "data/mini_test.csv" ), iter=3.2)
      Traceback (most recent call last):
        ...
      TypeError: iter must be integer
      >>> LearningAlg(pd.read_csv( "data/mini_test.csv" ), iter=-1)
      Traceback (most recent call last):
        ...
      ValueError: iter must be greater than zero
      >>> LearningAlg(pd.read_csv( "data/mini_test.csv" ), eps='a')
      Traceback (most recent call last):
        ...
      TypeError: eps and eta must be float
      >>> LearningAlg(pd.read_csv( "data/mini_test.csv" ), eps=0.6)
      Traceback (most recent call last):
        ...
      ValueError: eps and must be => 0 and < 0.5
    """
    if not isinstance( df, pd.DataFrame ):
        raise TypeError( "df must be pandas dataframe" )
    if type(iter) != int:
        raise TypeError( "iter must be integer" )
    if iter < 1:
        raise ValueError( "iter must be greater than zero" )
    if type(eps) != float or type(eta) != float:
        raise TypeError( "eps and eta must be float" )
    if not (0 <= eps < 0.5 and 0 <= eta < 0.5):
        raise ValueError( "eps and must be => 0 and < 0.5" )
    num = int(df.columns[len(df.columns) - 1].split(sep = 'x')[0])
    p_k = np.zeros( (2,) )
    P = np.zeros( (2, num, num) )
    p_k_x = np.zeros( (2, len( df )) )
    p_k_x[0] = np.random.uniform( eta, 1 - eta, size=len( df ) )
    p_k_x[1] = 1 - p_k_x[0]
    X = df.to_numpy()[:, np.argmax(df.columns == '1x1'):].T.reshape( (num, num, len( df )) )
    for i in range( iter ):
        p_k = np.sum( p_k_x, axis=1 ) / len( df )
        P[0] = np.sum( X * p_k_x[0], axis=2 ) / (np.sum( p_k_x[0] ) + (np.sum( p_k_x[0] ) == 0))
        P[1] = np.sum( X * p_k_x[1], axis=2 ) / (np.sum( p_k_x[1] ) + (np.sum( p_k_x[1] ) == 0))
        P_min = np.array( [P < eps] )[0]
        P_max = np.array( [P > 1 - eps] )[0]
        P = P * (1 - np.max( np.array( [P_min, P_max] ), axis=0 )) + (1 - eps) * P_max + eps * P_min
        nu = np.prod( np.prod( ((P[0] / P[1]) ** X.T) * (((1 - P[0]) / (1 - P[1])) ** (1 - X.T)), axis=1 ), axis=1 )
        p_k_x[0] = nu * p_k[0] / (p_k[1] + nu * p_k[0])
        p_k_x[1] = p_k[1] / (p_k[1] + nu * p_k[0])
    return P, p_k

if __name__ == "__main__":
    import doctest
    doctest.testmod()