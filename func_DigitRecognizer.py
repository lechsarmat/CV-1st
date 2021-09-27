import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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


def CheckAlg(df, P, p_k):
    """Checks performance of EM-algorithm on dataset.
       >>> CheckAlg(pd.read_csv( "data/mini_test.csv" ), np.array([np.zeros((4,4))+0.95,np.zeros((4,4))+0.05]), np.array([0.5, 0.5]))
       1.0
       >>> CheckAlg(0, np.array([np.zeros((4,4))+0.95,np.zeros((4,4))+0.05]), np.array([0.5, 0.5]))
       Traceback (most recent call last):
        ...
       TypeError: df must be pandas dataframe
       >>> CheckAlg(pd.read_csv( "data/mini_test.csv" ), 0, np.array([0.5, 0.5]))
       Traceback (most recent call last):
        ...
       TypeError: P and p_k must be numpy array
       >>> CheckAlg(pd.read_csv( "data/mini_test.csv" ), np.array([np.zeros((4,4))+0.95,np.zeros((4,4))+0.05]), np.array(['a', 0.5]))
       Traceback (most recent call last):
        ...
       TypeError: wrong type of elements of P or p_k
       >>> CheckAlg(pd.read_csv( "data/mini_test.csv" ), np.array([np.zeros((4,4))+0.95,np.zeros((4,4))+0.05]), np.array([0.5, 0.5, 0.5]))
       Traceback (most recent call last):
        ...
       ValueError: P must have two sub-matrices and p_k must have two values
       >>> CheckAlg(pd.read_csv( "data/mini_test.csv" ), np.array([np.zeros((4,4))+0.95,np.zeros((4,4))+0.05]), np.array([2., 0.5]))
       Traceback (most recent call last):
        ...
       ValueError: elements of P and p_k must be > 0 and < 1
       >>> CheckAlg(pd.read_csv( "data/mini_test.csv" ), np.array([np.zeros((5,5))+0.95,np.zeros((5,5))+0.05]), np.array([0.5, 0.5]))
       Traceback (most recent call last):
        ...
       ValueError: P must have the same shape as data
    """
    if not isinstance( df, pd.DataFrame ):
        raise TypeError( "df must be pandas dataframe" )
    if type( P ) != np.ndarray or type( p_k ) != np.ndarray:
        raise TypeError( "P and p_k must be numpy array" )
    if P.dtype != 'float64' or p_k.dtype != 'float64':
        raise TypeError( "wrong type of elements of P or p_k" )
    if P.shape[0] != 2 or p_k.shape[0] != 2:
        raise ValueError( "P must have two sub-matrices and p_k must have two values" )
    if not (np.prod(0. < P) * np.prod(P < 1.) and np.prod(0. < p_k) * np.prod(p_k < 1.)):
        raise ValueError( "elements of P and p_k must be > 0 and < 1" )
    num = int(df.columns[len( df.columns ) - 1].split( sep='x' )[0])
    if P.shape[1] != num or P.shape[2] != num:
        raise ValueError( "P must have the same shape as data" )
    X = df.to_numpy()[:, 1:].T.reshape( (num, num, len( df )) )
    p_x_k = np.zeros( (2, len( df )) )
    p_x_k[0] = np.prod( np.prod( (P[0] ** X.T) * ((1 - P[0]) ** (1 - X.T)), axis=1 ), axis=1 )
    p_x_k[1] = np.prod( np.prod( (P[1] ** X.T) * ((1 - P[1]) ** (1 - X.T)), axis=1 ), axis=1 )
    Pr = (p_x_k.T * p_k).T / (np.sum( p_x_k.T * p_k, axis=1 ) + np.array( [np.sum( p_x_k.T * p_k, axis=1 ) == 0] ))
    Guess = np.array( [0 if Pr[0][i] > Pr[1][i] else 1 for i in range( len( df ) )] )
    Q = np.sum( np.array( [Guess == df.label] )[0] ) / len( df )
    return max( Q, 1 - Q )


def ShowPlot(P_k):
    """Displays a graphical representation of probability matrices.
       >>> ShowPlot(0)
       Traceback (most recent call last):
        ...
       TypeError: P_k must be numpy array
       >>> ShowPlot(np.array(['a',0]))
       Traceback (most recent call last):
        ...
       TypeError: wrong type of elements
       >>> ShowPlot(np.array([0.]))
       Traceback (most recent call last):
        ...
       ValueError: P_k must have at least two sub-matrices
    """
    if type(P_k) != np.ndarray:
        raise TypeError( "P_k must be numpy array" )
    if P_k.dtype != 'float64':
        raise TypeError( "wrong type of elements" )
    if P_k.shape[0] < 2:
        raise ValueError( "P_k must have at least two sub-matrices" )
    fig = plt.figure( figsize=(16, 8) )
    ax1 = fig.add_subplot( 121 )
    ax1.imshow( P_k[0], cmap=plt.get_cmap( 'gray' ) )
    ax1.set_title( 'Class 0' )
    ax2 = fig.add_subplot( 122 )
    ax2.imshow( P_k[1], cmap=plt.get_cmap( 'gray' ) )
    ax2.set_title( 'Class 1' )
    plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()