class GeneticAlgorithm:

    def __init__(self, population_size, n_epochs):
        self.population_size = population_size
        self.n_epochs = n_epochs
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train, stop_after_no_progress):
        self.X_train = X_train
        self.y_train = y_train


        # generate starting population

        # find best individual to see progress

        '''
        repeat for num_epochs
            generate new population with selection, crossover and mutation
            find best individual
            compare it with current best individuals
            exit with stopping condition (>num_epoch, >epoch_no_progress
            
        return all best individuals 
            
        '''
