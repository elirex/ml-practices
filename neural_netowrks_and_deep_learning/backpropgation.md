# Backpropgation

### [Backprogpation derivation detail](http://cpmarkchang.logdown.com/posts/277349-neural-network-backward-propagation)
![](backpropgation.png)

### Basic idea

1. Derivative

    ![](backpropgation_derivative.png)

    Blue line: ![](https://latex.codecogs.com/svg.latex?y%20%3D%20x%5E%7B2%7D)

    Green line: ![](https://latex.codecogs.com/svg.latex?%7Bf%7D%27%28x%29%3D2x)

    <b>Formula:</b> ![](https://latex.codecogs.com/svg.latex?%5Cfrac%7Bd%7D%7Bdx%7Dx%5E%7Bn%7D%20%3D%20nx%5E%7Bn-1%7D)

2. Partial Derivative

    Original function: ![](https://latex.codecogs.com/svg.latex?f%28x%2C%20y%29%3Dy%5E%7B4%7D&plus;5xy)

    Partial derivate with respect to `x`: ![](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x%7D%3D5y)

    Partial derivate with respect to `y`: ![](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20y%7D%3D%204y%5E%7B3%7D%20&plus;%205x)

3. The Chain Rule

    If ![](https://latex.codecogs.com/svg.latex?f%28x%29%3D2x%20%5C%20and%20%5C%20g%28x%29%3Dx%5E%7B2%7D), then ![](https://latex.codecogs.com/svg.latex?f%28g%28x%29%29%3D2%28x%5E%7B2%7D%29), what's it's derivative?
    
    The chain rule: ![](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20%7D%7B%5Cpartial%20x%7D%5Bf%28g%28x%29%29%5D%20%3D%20%7Bf%7D%27%28g%28x%29%29g%5E%7Bx%7D)

    Result: ![](https://latex.codecogs.com/svg.latex?%7Bf%7D%27%28x%29%3D2%20%5C%20and%20%5C%20%7Bg%7D%27%28x%29%3D2x%2C%20so%20%5C%20%7Bf%7D%27%28%7Bg%7D%27%28x%29%29%20%3D%204)

<b>Formual:</b>

![](https://latex.codecogs.com/svg.latex?%7B%28h%28x%29%29%29%7D%27%3D%7Bg%28f%28x%29%29%7D%27%3D%7Bg%7D%27%28f%28x%29%29%7Bf%28x%29%7D%27)

### References

* <a>http://cpmarkchang.logdown.com/posts/277349-neural-network-backward-propagation</a>
* <a>https://www.youtube.com/watch?v=q555kfIFUCM</a>