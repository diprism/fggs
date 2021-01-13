# It might be worth having subclasses: one for finite variable domains, one for infinite.
# These subclasses could then implement sum_product and the sampling methods differently.


from abc import ABC, abstractmethod

class FGG(ABC):

    # IO Functions
    @abstractmethod
    def load_from_file(self, filename):
        pass
    
    @abstractmethod
    def print_to_file(self, filename):
        pass
    

    # Conjunction
    @abstractmethod
    def conjoin(self, f):
        # conjoin with another FGG, return new FGG as result
        pass


    # Sampling
    @abstractmethod
    def sample(self):
        # sample a graph structure + instantions of the variables
        pass
    
    @abstractmethod
    def sample_graph(self):
        # sample just the graph structure
        pass
    
    
    # Inference
    @abstractmethod
    def sum_product(self):
        # compute the sum-product
        pass
    
    @abstractmethod
    def viterbi(self):
        # find the most probable graph + setting of variables
        pass
    
    
    # Learning
    @abstractmethod
    def learn_weights(self, training_data):
        # learn the weights for the factor functions
        pass
        
    
    # Not sure if we want/need this stuff:
    @abstractmethod
    def parse_graph(self, graph):
        # parse a graph using the grammar
        pass
