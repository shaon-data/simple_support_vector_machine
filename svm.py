## commit
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

plus_class = 1
minus_class = -1

## for avoiding retraining we are making class for object creation
class svm:
    def __init__(self, visualize=True):
        self.visualize = visualize
        self.colors = {plus_class:'r',minus_class:'b'}
        
        if self.visualize:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

        ## train
        def fit(self,data):
            self.data = data
            ## { ||w||: [w,b] }
            mag = {}

            transforms =  [[1,1],
                           [-1,1],
                           [-1,-1],
                           [1,-1]]
            
        def predict(self,features):
            # sign(x.w+b)
            classify = np.sign( np.dot(np.array(features),self.w) + self.b )
            return classify
            
data_dict = {
              plus_class:np.array([[1,7],
                                   [2,8],
                                   [3,8]
                                  ]),
              
              minus_class:np.array([[5,1],
                                    [6,-1],
                                    [7,3]
                                  ])
            }
