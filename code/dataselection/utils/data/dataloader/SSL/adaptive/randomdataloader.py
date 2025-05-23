from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionmethods.SL import RandomStrategy
import time, copy, logging

class RandomDataLoader(AdaptiveDSSDataLoader):
    """
    Implements of RandomDataLoader that serves as the dataloader for the non-adaptive Random subset selection strategy.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    dss_args: dict
        Data subset selection arguments dictionary required for Random subset selection strategy
    logger: class
        Logger for logging the information
    """
    def __init__(self, train_loader, dss_args, logger, *args, **kwargs):
        """
        Constructor function
        """
        super(RandomDataLoader, self).__init__(train_loader, train_loader, dss_args, 
                                               logger, *args, **kwargs)
        self.strategy = RandomStrategy(train_loader, online=False)
        self.logger.debug('Random dataloader initialized. ')

    def _resample_subset_indices(self):
        """
        Function that calls the Random subset selection strategy to sample new subset indices and the corresponding subset weights.
        """
        start = time.time()
        self.logger.debug("Iteration: {0:d}, requires subset selection. ".format(self.cur_iter))
        logging.debug("Random budget: %d", self.budget)
        subset_indices, _ = self.strategy.select(self.budget)
        end = time.time()
        self.logger.info("Iteration: {0:d}, subset selection finished, takes {1:.2f}. ".format(self.cur_iter, (end - start)))
        return subset_indices
