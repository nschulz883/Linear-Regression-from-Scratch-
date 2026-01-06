#linear regression from scratch which is ML adjacent
#do linear calculations then compare with expected 
#result

#finished but add more linear analysis if bored 

# training data 

x = [1,2,4,8,16]
y = [2,6,8,16,30]

x_min = min(x)
x_max = max(x)
x = [(v - x_min) / (x_max - x_min) for v in x]

max_grad= 2


c = 0
m = 2
N = len(x)

learning_rate = 0.1 
epochs = 1000





#predicted value calculations   y = mx + c 

ypred = []
yerror = []
loss_history = []
# whole code where stuff is executed goes in epoch loop

for epoch in range(epochs):
  
    ypred = []
    yerror = []
    loss = []
  
    i = 0
    for i in range (N):
        ypred.append(m * x[i] + c)     
       
    dm = 0.0
    dc = 0.0

          
    for a in range (N):
        yerror.append(ypred[a] - y[a])
        dm += x[a] * yerror[a]
        dc += yerror[a]


    dm = (2/N) * dm 
    dc = (2/N) * dc

    #  clip gradients prevents crazy numbers
    
    if dm > max_grad:
        dm = max_grad 
    if dc > max_grad:
        dc = max_grad

    dm = max(-max_grad, min(max_grad, dm))
    dc = max(-max_grad, min(max_grad, dc))

    m -= learning_rate * dm
    c -= learning_rate * dc 


    #mean square average calculation this tracks learning over time
    loss = sum((ypred[i] - y[i])**2 for i in range(N)) / N
    loss_history.append(loss)
    
    #shows improvement per 100 epochs
    if epoch % 100 == 0:
         print("Epoch {}: Loss = {:.2f}, m = {:.4f}, c = {:.4f}".format(epoch, loss, m, c))

  #data display 
for i in range (N): 
    print("X value:" , x[i]) 
    print("actual Y value:", y[i])
    print("predicted Y value:", m * x[i] + c) 
    print("Difference:", (m * x[i] + c - y[i]  ))  
    print("residual:", i, "=",ypred[i]-y[i])
    print()
