%% Machine Learning Online Class
%  Exercise 8 | Anomaly Detection and Collaborative Filtering
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     estimateGaussian.m
%     selectThreshold.m
%     cofiCostFunc.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% =============== Part 1: Loading movie ratings dataset ================
%  You will start by loading the movie ratings dataset to understand the
%  structure of the data.
%  
fprintf('Loading movie ratings dataset.\n\n');

%  Load data
load ('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  From the matrix, we can compute statistics like average rating.
fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', ...
        mean(Y(1, R(1, :))));

%  We can "visualize" the ratings matrix by plotting it with imagesc
imagesc(Y);
ylabel('Movies');
xlabel('Users');

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% ============ Part 2: Collaborative Filtering Cost Function ===========
%  You will now implement the cost function for collaborative filtering.
%  To help you debug your cost function, we have included set of weights
%  that we trained on that. Specifically, you should complete the code in 
%  cofiCostFunc.m to return J.

%  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
load ('ex8_movieParams.mat');

%  Reduce the data set size so that this runs faster
num_users = 4; num_movies = 5; num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

%  Evaluate cost function
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
               num_features, 0);
           
fprintf(['Cost at loaded parameters: %f '...
         '\n(this value should be about 22.22)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% ============== Part 3: Collaborative Filtering Gradient ==============
%  Once your cost function matches up with ours, you should now implement 
%  the collaborative filtering gradient function. Specifically, you should 
%  complete the code in cofiCostFunc.m to return the grad argument.
%  
fprintf('\nChecking Gradients (without regularization) ... \n');

%  Check gradients by running checkNNGradients
%checkCostFunction;

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% ========= Part 4: Collaborative Filtering Cost Regularization ========
%  Now, you should implement regularization for the cost function for 
%  collaborative filtering. You can implement it by adding the cost of
%  regularization to the original cost computation.
%  

%  Evaluate cost function
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
               num_features, 1.5);
           
fprintf(['Cost at loaded parameters (lambda = 1.5): %f '...
         '\n(this value should be about 31.34)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% ======= Part 5: Collaborative Filtering Gradient Regularization ======
%  Once your cost matches up with ours, you should proceed to implement 
%  regularization for the gradient. 
%

%  
fprintf('\nChecking Gradients (with regularization) ... \n');

%  Check gradients by running checkNNGradients
%checkCostFunction(1.5);

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% ============== Part 6: Entering ratings for a new user ===============
%  Before we will train the collaborative filtering model, we will first
%  add ratings that correspond to a new user that we just observed. This
%  part of the code will also allow you to put in your own ratings for the
%  movies in our dataset!
%
movieList = loadMovieList();

%  Initialize my ratings
my_ratings = zeros(1682, 1);

% Check the file movie_idx.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
%my_ratings(1) = 4;

% Or suppose did not enjoy Silence of the Lambs (1991), you can set
%my_ratings(98) = 2;

% We have selected a few movies we liked / did not like and the ratings we
% gave are as follows:
%my_ratings(7) = 3;
%my_ratings(12)= 5;
%my_ratings(54) = 4;
%my_ratings(64)= 5;
%my_ratings(66)= 3;
%my_ratings(69) = 5;
%my_ratings(183) = 4;
%my_ratings(226) = 5;
%my_ratings(355)= 5;

my_ratings(109) = 1; %Mystery Science Theater 3000: The Movie (1996)
my_ratings(120) = 1; %Striptease (1996)
my_ratings(122) = 1; %Cable Guy, The (1996)
my_ratings(145) = 1; %Lawnmower Man, The (1992)
my_ratings(5) = 2; %Copycat (1995)
my_ratings(179) = 2; %Clockwork Orange, A (1971)
my_ratings(187) = 2; %Godfather: Part II, The (1974)
my_ratings(17) = 3; %From Dusk Till Dawn (1996)
my_ratings(29) = 3; %Batman Forever (1995)
my_ratings(67) = 3; %Ace Ventura: Pet Detective (1994)
my_ratings(125) = 3; %Phenomenon (1996)
my_ratings(156) = 3; %Reservoir Dogs (1992)
my_ratings(161) = 3; %Top Gun (1986)
my_ratings(235) = 3; %Mars Attacks! (1996)
my_ratings(1) = 4; %Toy Story (1995)
my_ratings(2) = 4; %GoldenEye (1995)
my_ratings(11) = 4; %Seven (Se7en) (1995)
my_ratings(23) = 4; %Taxi Driver (1976)
my_ratings(28) = 4; %Apollo 13 (1995)
my_ratings(55) = 4; %Professional, The (1994)
my_ratings(71) = 4; %Lion King, The (1994)
my_ratings(72) = 4; %Mask, The (1994)
my_ratings(94) = 4; %Home Alone (1990)
my_ratings(117) = 4; %Rock, The (1996)
my_ratings(127) = 4; %Godfather, The (1972)
my_ratings(163) = 4; %Return of the Pink Panther, The (1974)
my_ratings(198) = 4; %Nikita (La Femme Nikita) (1990)
my_ratings(231) = 4; %Batman Returns (1992)
my_ratings(7) = 5; %Twelve Monkeys (1995)
my_ratings(22) = 5; %Braveheart (1995)
my_ratings(24) = 5; %Rumble in the Bronx (1995)
my_ratings(50) = 5; %Star Wars (1977)
my_ratings(56) = 5; %Pulp Fiction (1994)
my_ratings(62) = 5; %Stargate (1994)
my_ratings(69) = 5; %Forrest Gump (1994)
my_ratings(80) = 5; %Hot Shots! Part Deux (1993)
my_ratings(82) = 5; %Jurassic Park (1993)
my_ratings(89) = 5; %Blade Runner (1982)
my_ratings(95) = 5; %Aladdin (1992)
my_ratings(96) = 5; %Terminator 2: Judgment Day (1991)
my_ratings(98) = 5; %Silence of the Lambs, The (1991)
my_ratings(121) = 5; %Independence Day (ID4) (1996)
my_ratings(144) = 5; %Die Hard (1988)
my_ratings(172) = 5; %Empire Strikes Back, The (1980)
my_ratings(174) = 5; %Raiders of the Lost Ark (1981)
my_ratings(176) = 5; %Aliens (1986)
my_ratings(181) = 5; %Return of the Jedi (1983)
my_ratings(182) = 5; %GoodFellas (1990)
my_ratings(183) = 5; %Alien (1979)
my_ratings(195) = 5; %Terminator, The (1984)
my_ratings(196) = 5; %Dead Poets Society (1989)
my_ratings(202) = 5; %Groundhog Day (1993)
my_ratings(204) = 5; %Back to the Future (1985)
my_ratings(210) = 5; %Indiana Jones and the Last Crusade (1989)
my_ratings(217) = 5; %Bram Stoker's Dracula (1992)
my_ratings(226) = 5; %Die Hard 2 (1990)

fprintf('\n\nNew user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% ================== Part 7: Learning Movie Ratings ====================
%  Now, you will train the collaborative filtering model on a movie rating 
%  dataset of 1682 movies and 943 users
%

fprintf('\nTraining collaborative filtering...\n');

%  Load data
load('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 10;
theta = fmincg (@(t)(cofiCostFunc(t, Y, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

fprintf('Recommender system learning completed.\n');

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% ================== Part 8: Recommendation for you ====================
%  After training the model, you can now make recommendations by computing
%  the predictions matrix.
%

p = X * Theta';
my_predictions = p(:,1) + Ymean;

movieList = loadMovieList();

[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for you:\n');
count = 0
for i=1:length(ix)
    j = ix(i);
    if my_ratings(j) == 0 && my_predictions(j) > 7.0
      
      fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), ...
              movieList{j});
      count = count + 1;
    end          
end

%fprintf('\n\nOriginal ratings provided:\n');
%for i = 1:length(my_ratings)
%    if my_ratings(i) > 0 
%        fprintf('Rated %d for %s\n', my_ratings(i), ...
%                 movieList{i});
%    end
%end
