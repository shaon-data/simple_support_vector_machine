## commit
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

plus_class = 1
minus_class = -1

## for avoiding retraining we are making class for object creation
class support_vector_machine:
    def __init__(self, visualize=True):
        self.visualize = visualize
        self.colors = {plus_class:'r',minus_class:'b'}
        if self.visualize:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    ## train
    def fit(self, data):
        self.data = data
        ## { ||w||: [w,b] }
        mag = {}

        ## testing all signs because square does not affect but dot product does
        transforms =  [[1,1],
                       [-1,1],
                       [-1,-1],
                       [1,-1]]
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature  in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      ## point of expense:
                      self.max_feature_value * 0.001,]

        ## extremely expensive
        b_range_multiple = 5

        ## we don't need to take as small of steps
        ## with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value*10
                
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            ## we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                        self.max_feature_value*b_range_multiple,
                                        step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        ## weakest link in the SVM fundamentally
                        ## SMO attempts to fix this a bit
                        ## constraint equation yi(xi.w+b) >= 1
                        #
                        ## #### add a break here later....
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                            
                        if found_option:
                            mag[np.linalg.norm(w_t)] = [w_t,b] ## norm - magnitude of a vector

                if w[0] < 0:
                    optimized = True
                    print("optimized a step.")
                else:
                    w = w - step
                            
            norms = sorted([n for n in mag])
            ## ||w|| : [w,b]
            opt_choice = mag[norms[0]]
            self.w=opt_choice[0]
            self.b=opt_choice[1]
            latest_optimum = opt_choice[0][0] + step*2
                    
    def predict(self,features):
        # sign(x.w+b)
        classify = np.sign( np.dot(np.array(features),self.w) + self.b )
        if classify != 0 and self.visualization:
            self.ax.scatter(feature[0],feature[1],s=200,marker='*',c=self.colors[classify])
        return classify
        
    def visualize_g(self):
        [[self.ax.scatter(x[0] , x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane x.w + b
        ## v = x.w + b
        ## psv = 1
        ## nsv = -1
        ## dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x - b + v) /w[1] # i

        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        ## (w.x+b) = 1
        ## positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2])

        ## (w.x+b) = -1
        ## positive support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2])

        ## (w.x+b) = 0
        ## decision boundary support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2])

        plt.show()
                
                
            
data_dict = {
              minus_class:np.array([[1,7],
                                   [2,8],
                                   [3,8],
                                  ]),
              
              plus_class:np.array([[5,1],
                                   [6,-1],
                                   [7,3],
                                  ])
            }

svm = support_vector_machine()
svm.fit(data=data_dict)
svm.visualize_g()
