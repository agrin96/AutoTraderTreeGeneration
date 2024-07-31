# Trading Tree Generator

This project intended to create dual-tree models which can be used for cryptocurrency autotrading. 

Model 1 would be a `buy` model which decides when a good entry point is and model 2 would be a `sell` model which takes over to decide when a good exit for the position is. The dual model architecture was intended to be better than one model for everything since it can be tuned to look for one particular signal rather than trying to look for both buy and sell signals. Each model is a tree model trained using a custom implementation of a genetic algorithm. Implementing the genetic algorithm myself allowed more granular tuning of the model training process.

# Project Status (Dead)

Just as the Binance auto trader itself, this project is complete and dead. The models generated were never too successful at their intended tasks, but the exercise was quite cool and I still believe the architecture has merit. After a lot of training and learning I no longer believe that simple price data can be used to train a bot that is more effective than buy and hold. I may be wrong of course, but the problem seems to difficult for the approach chosen.
