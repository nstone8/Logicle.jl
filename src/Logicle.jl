module Logicle
using Optim, Statistics


export LogicleScale

#Calculate logicle transform as described in https://doi.org/10.1002/cyto.a.22030

"""
```julia
LogicleScale(T,W,M,A)
```
Build a `LogicleScale` as described in [https://doi.org/10.1002/cyto.a.22030] from parameters.
The scale can then be used to transform a value by calling the struct as a function

# Example
```julia
l = LogicleScale(T,W,M,A)
#if x is some iterable
x_scaled = l.(x)
```
"""
struct LogicleScale
    a::Real
    b::Real
    c::Real
    d::Real
    f::Real

    function LogicleScale(T::Real,W::Real,M::Real,A::Real)
        #compute our parameters from arguments
        b = (M + A)*log(10)
        w = W/(M+A)
        x2 = A/(M+A)
        x1 = x2 + w
        x0 = x1 + w

        #define g(d) which should be zero for our desired value of d
        #optimize expects arrays, so d will be a length 1 array
        #wrap in abs because we want the root, not the minimum
        g(d) = abs(w*(b+d[1]) + 2*(log(d[1]) - log(b)))
        #optimize, use an initial guess of b/2
        dopt = optimize(g,[0],[b],[b/2];autodiff = :forward)
        #check that we converged
        @assert Optim.converged(dopt) "didn't converge on a value for d"
        d = Optim.minimizer(dopt)[1]
        covera = exp((b+d)*x0)
        fovera = covera*exp(-d*x1)-exp(b*x1)
        a = T/(exp(b) - covera*exp(-d) + fovera)
        c = covera*a
        f = fovera*a
        new(a,b,c,d,f)
    end
    
end

"""
```julia
inv_scale(l,y)
```
Given a scaled value `y` which has been scaled according to the `LogisticScale` `l`, return the
unscaled data value
"""
function inv_scale(l::LogicleScale,y::Real)
    l.a*exp(l.b*y) - l.c*exp(-l.d*y) + l.f
end

function (l::LogicleScale)(x::Real)
    #our scaled value y is defined in terms of the inverse function
    #i.e inv_scale(l,y) = x
    #0 = x - inv_scale(l,y)
    resid(y) = abs(x - inv_scale(l,y[1]))
    yopt = optimize(resid ,[0.0],autodiff=:forward)
    @assert Optim.converged(yopt) "didn't converge on a value for y"
    Optim.minimizer(yopt)[1]
end

"""
```julia
LogicleScale(data;[T,W,M,A])
```
Build a `LogicleScale`. Estimate the parameter `T` and `W` from the data and use
default values for `M` and `A` if not provided.
"""
function LogicleScale(data;T=nothing,W=nothing,M=4.5,A=0)
    T = isnothing(T) ? maximum(data) : T
    W = isnothing(W) ? begin
        negvals = filter(data) do d
            d<0
        end
        if isempty(negvals)
            #provided data contains no negative values. W=0 will make this more like a
            #normal log scale
            0
        else
            r = quantile(negvals,.05)
            (M - log10(T/abs(r)))/2
        end
    end : W
    LogicleScale(T,W,M,A)
end

end # module Logicle
