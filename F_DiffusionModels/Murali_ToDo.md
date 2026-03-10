0. Probability Distribution & KL Divergence.

1. Auto Encoders (Deepia - YT channel)
    a. Latent Space
    b. Latent Dimension (How it impacts the reconstruction quality)
    c. UNET - Model for image processing tasks.
    d. How do we Train Auto Encoder Network.
        - Encoders - does dimensionality reduction Preserving Important Info.
            - A good Encoder - after dimensionality reduction will be able to make uniquely identifiable clusters in the latent space per target class.
            - How Latent Space Evolves during training(with increasing Epochs).

        - Decoders - trained to reconstruct accurately.
        - Loss function => Mean squared error(MSE) of Original input vs Regenerated output.
        - Optimize Loss function.
    e. Limitations of Auto encoders
        - If we take 2 points and reconstruct  from a middle point(b/w the reference 2 points) - we expect resconstruction from this middle point to be mixture of reference points. But unfortunately this is not the case.
        - It learns many irrelevent features.
    f. So Regularize latent Space  => Variational Auto encoders.

2. Dimensionality Reduction
    a. PCA
        - say we have 2D data. each dimension is a Principle Componets.
        - Project data on largest Principal component.
            - WHy we do not project data on Horizontal axis or Vertical axis?
            - So Projecting on Principle Component is better.
        - How we calculate Principle components
            - Covarience matrix of dataset.
            - Find Eigen vectors of Covariance matrix. Eigen vectors with largest Eigen values are our Principle components
        - Advantages of PCA (Most fastest and most interpretable method)
            - It relies basically on Linear Algebra - so it is very Fast.
            - Its Time complexity formulae. (Time will increase proportionally with data set size)
            - Visualizing Latentspace and Principle components through Eigen vectors is very easy.
        - Limitations
            - It does not work well with non linear data. Example - Spiral data.

    b. t-SNE
        - What is SNE?
            - Idea is - Points which are close to each other in Higher dimensional space should also be close to each other in its Latent space(Lower dimensional space)
                - To make this happen - we measure distance bw point and its neighbours in high dimensional space. and ensure those distances stay similar in Lower dimensional space
                - for this we use speacial kind of Loss function. Loss(HighDim, LowDim) and Minimize loss.
            - Steps
                1. Turn distance b/w point and its neighbours as Probability distribution - specifically to gaussian distribution? Ans=> What is probability of Blue point to be neighbour of a red point.
                2. Formulae to get probability of each point in high dimension space  
                  P(0,j) = E^(-(||x0-xj||^2)/2sigma^2)
                3. sigma - ALso varience used to change the shape of (gaussian)Bell curve.
                4. shape of the bell curve is crucial - because 
                        - if its wider, more points will be considered as neighbours
                        - if its narrow only closest points will be considered as neighbours.
                5. Perplexity - We adjust variance of this gaussian using hyperparameter called perplexity.
                6. To make all these probalilities add up to 1, we normalize each gaussian by sum of all the others. Formulae?
                    But this is very costly - in terms of computation.
                6. Repeat same process (steps 1 to 5) for each point in High dimensional space 
                7. In our Lower dimensional space - start by placing points randomly
                    repeat the same process as we did with High dimensional data.(steps 1 to 6, but for Low dimension space points and not High Dimension space points)
                8. say q(i,j) is probability distribution of Low dimenson space and p(i,j) is probability distribution of High dimenson space
                9. Bring p(i,j) and q(i,j) - closer to each other.
                10. KL divergence => Formulae? 
                    - as P and Q are apart KL divergence increases.
                    - as Pand Q become similar - KL divergence decreases.
                    - if P and Q match perfectly - KL divergence is 0.
                11. Next calculate - gradient of KL divergence with Low dimensional embedding. Formulae?
                12. Now we can adjust the Low dimensional representation, to ensure it is closer to original distribution.
                13. Provide a Demo (how this evolves with each epoch of training).
            - What is Perplexity hyperparameter?
                - Perplexity helps us choose - correct sigma parameter which controls the width of the gaussian in high dimensional representation.
                - How to find it? Calculated using entropy of each distribution using this formulae => P =  2^(H(pi)), where H(pi)=-Summation(p(i,j)log(p(i,j)))
                - In practice - ALgo will choose diffrent values of sigma - untill the above formulae is equal to perplexity chosen by user.
                - Perplexity is  number of other points - a point considers as neighbours. But it is not so obvious.
                - Visualize - what happens when you increase perplexity.
        - Advantages of SNE => works with Non Linear data (demo using spiral data) compare PCA vs SNE for this spiral data.
        - Limitations of SNE => It is terribly slower.
        - t-SNE ?
            - using t distribution in low dimension space. What is t-distribution?
            - these distributions are easier to compute as they do not use the exponential function.
        - SLOW? even with many modifications and hacks, t-sne is very slow compared to PCA, perplexity(if changed little) - will mess up our visualization, means we need to check visualization each time we change perplexity - untill we get good representation.



    c. UMAP and variations of UMAP. (Here Understand K Nearest Neighbours Algo)
        - 2018 - UMAP was introduced. 
        - UMAP - Uniform Manifold Approximation and Projection.
        - Steps of Algo. (somewhat similar to t-SNE)
            - Look at distance to each neighbour of every high dimensional sample, and aim for Low dimensional representaion that roughly matches those distances.
            - Here instead of using Gaussian and Probability distributions, use Graphs to represent both High dimensional and Low dimensional data.
            - Find K - nearesrt Neighbours of each sample. (Here Understand K Nearest Neighbours Algo). and connect point to point to form a Graph.
                - K is main hyper parameter of UMAP algo.
                - This gives us a First binary graph.
            - Next transform this binary graph as weighted graph to represent - how close this point is to its neighbours.
                - Weight computation - is done by applying exponentially decay over distances between this point and its neighbours.
                - pi => we offset this decay by  distance of nearest neighbour so that closest neighbour keeps a weight of 1. we have similar formulae for t-SNE - but we do not normalize this quantity, making it much faster to compute.
            - Now repeat the above step for all other samples.
                - After this we - will endup with 1 weighted graph per sample. => which we need to combine to single weighted graph.
                    - How to combine? Formulaae to combining weighted binary graph? 
                    - with weighted binary hraph?. steps below
                    `   -  for this ensure there is only 1 edge b/w any two points instead of 2. w(i,j)= v(i,j) + v(j,i) - v(i,j).v(j,i)
                    - Now we have final graph.
            - Repeat same for Low Dimensional space - points.
            - Next Step?  
                - Try to match  Both graphs (Low Dimension graph vs High Dim graph) -  with help of a Loss function.
                - Get Adjacency matrices (using graphs) for both Low & High Dimension graphs
                - Compute the Cross entropy over  these matrices and simply minimize it using stochastic gradient descent and adjust the Low dimensional representation accordingly.
            - UMAP Pros
                - behoviour is much more predictable with changing hyper parameter (Perplexity)
                - It prserves features in 2D space will be similar to that of 3D projection.
    d. Others - Trimap and PacMap

3. Variational Auto Encoders
    - 