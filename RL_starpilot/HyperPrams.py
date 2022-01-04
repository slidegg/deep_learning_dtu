
# Edit the hyper parameters in the config.yml

class HyperPrams():

    def __init__(self, gamma, lmbda, learning_rate, grad_clip_norm, eps_clip, value_coef, entropy_coef) -> None:
        
        self.gamma = gamma;
        self.lmbda = lmbda; 
        self.learning_rate = learning_rate; 
        self.grad_clip_norm = grad_clip_norm; 
        self.eps_clip = eps_clip; 
        self.value_coef = value_coef; 
        self.entropy_coef = entropy_coef; 
    

