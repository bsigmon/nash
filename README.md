# Search for Nash Equilibrium
I wanted to be able to find the nash equilibrium and thought I could perhaps use gradient descent, but quickly realized this would not easily work as I have two goals, one for each player, rather than a single cost function.

I then found a paper on [Competetive Gradient Descent](https://f-t-s.github.io/projects/cgd/) and implemented it.  It seemed promising, but ultimately, it would not converge to a clear solution.

## The Ace to 5 betting game
In order to test the algorithm, I implemented it for a problem with a known solution.

In [Play Optimal Poker](https://www.amazon.com/Play-Optimal-Poker-Practical-Theory-ebook/dp/B07SGGC53Q/),
by Andrew Brokos, he describes a simplified betting game which he calls "The Ace-to-Five Game"
> Two players, Opal and Ivan, are dealt a single card from a ten-card deck consisting of one of each card from 5 to A. Each player antes 1 chip, and the game permits only a single, 1 chip bet. That is, Opal may either bet 1 chip or check. Facing a check, Ivan may bet 1 chip or check. There is no raising, so facing a bet, a player's only options are to call or fold. If the hand goes to showdown, the player with the highest card wins.

I've attempted to compute the Nash equilibrium using [Competitive Gradient Descent](https://f-t-s.github.io/projects/cgd/) and have
left a [jupyter notbook implementation](https://www.kaggle.com/code/bsigmon/a-to-5-betting-game/edit) on [Kaggle](https://www.kaggle.com/) showing the results of this attempt.  Overall, I don't think it performed well.

## Alternative approach
From a starting point, pick a bunch (dozens, hundreds?) of points "nearby" and then do a min-max calculation to find a better solution.  That is, imagine a matrix with OPAL's strategies for columns and IVAN's strategies for rows, for each row, find the column with the maximal cost and then find the row with the minimal maximum.

This alone seems to find a local solution if I repeat and gradually decrease the distance of the nearby strategy points.  But, changing the RNG seed finds different solutions.  I then run the min-max over these solutions.

One can iterate this process a few times.

For the Ace to 5 game, I did three iterations of 250 nearby strategies with each of 100 different RNG seeds.  I ran the code with the 100 different RNGs somewhat in parallel using 5 worker threads.  I implemented this in rust and I'm pretty happy with the results.
