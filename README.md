# Fanorona Neural Network Training with GA and GP

This project focuses on training a neural network to play a simplified version of the Fanorona board game (5x5 instead of 5x9). The training process utilizes Genetic Algorithms (GA) and Genetic Programming (GP) to optimize the neural network's performance. 

## Methodology

1. **Genetic Algorithm (GA)**:  
   GA is used to optimize the neural network's weights by minimizing the loss rate, where fitness reflects the loss rate (lower is better). Mutations are performed by adding Gaussian noise to the weights, and the mutation rate is dynamically adjusted over generations. The crossover operation is not static but is dynamically evolved using GP to improve the fitness of the population. The implementation details can be found in `main_note.ipynb`.

2. **Genetic Programming (GP)**:  
   GP is employed to evolve the crossover operator dynamically. This allows the crossover operation to adapt and improve over time, leading to better offspring in the population. The GP uses a tree-based representation of operations and terminals to combine parent weights effectively. The implementation details can be found in `main_note.ipynb`.

3. **Training Process**:  
   - The training was conducted on Kaggle over 69 generations.  
   - Initially, training was performed on incomplete games (up to generation 44), and later on complete games.  
   - Fitness was calculated based on the loss rate across multiple games to account for the variability in gameplay outcomes.  
   - The training process and code are detailed in `main_note.ipynb`.

4. **Evaluation**:  
   The trained neural network was evaluated by playing against a random agent. The results showed a consistent decrease in the average fitness and standard deviation of the population, indicating the effectiveness of the GA. However, due to the stochastic nature of the games, the fitness reduction was not smooth but occurred in jumps. The evaluation process is also described in `main_note.ipynb`.

## Results

- The final model achieved the best performance in terms of minimizing the loss rate.  
- The statistics notebook (`statistics.ipynb`) demonstrates the reduction in average fitness and standard deviation over generations, highlighting the success of the GA.  
- The trained model was integrated into a Pygame-based game (`game_op.py`), allowing users to play against the neural network. The game represents a simplified version of Fanorona with a 5x5 board.

## Limitations

- The absence of a fixed dataset for training introduces variability in fitness calculations, as the games are not identical each time.  
- To mitigate this, a large number of games were used to compute fitness, ensuring more reliable results.

## Conclusion

This project demonstrates the successful application of GA and GP for training a neural network to play a complex board game. The dynamic evolution of the crossover operator using GP and the adaptive mutation strategy in GA were key factors in achieving the results. For further details on the implementation, refer to `main_note.ipynb`.
