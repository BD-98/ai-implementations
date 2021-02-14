using Flux, Statistics 
using Flux.Data: DataLoader
using  Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs 
using MLDatasets, CUDA
# Data Preprocessing 
xtr, ytr = MLDatasets.MNIST.traindata(Float32)
xtr = flatten(xtr)
ytr = onehotbatch(ytr, 0:9)
train_data = DataLoader(xtr, ytr, batchsize=32, shuffle=true)

# Model 
m = Chain(
    Dense(784, 128, relu),
    Dense(128, 10))

function calc_loss(loader, model)
    l = 0
    for (x, y) in loader
        l += logitcrossentropy(m(x), y)
    end
    return l / length(loader)
end

function acc(loader, model)
    accy = 0  
    for (x,y) in loader
        accy += mean(onecold(model(x)) .== onecold(y))
    end
    return accy 
end 

# Training 
loss(x, y) = logitcrossentropy(m(x), y)
evalcb() = @show calc_loss(train_data, m)
opt = ADAM(0.01)
@epochs 3 Flux.train!(loss, params(m), train_data, opt, cb = evalcb)
@show acc(train_data, m)

