
torch = require 'torch'
autograd = require 'autograd'
dist = require 'distributions'

shuffled_index = torch.randperm(200) 
-- print(shuffled_index) -- 1 to 200

mean = torch.Tensor({-5, -5})
covariance = torch.eye(2)

x_in_order = torch.Tensor(2,200)
y_in_order = torch.zeros(200)

for i = 1,100 do
	local a_sample = distributions.mvn.rnd(mean, covariance)
	x_in_order[1][i] = a_sample[1]
	x_in_order[2][i] = a_sample[2]
        y_in_order[i] = -1
end

mean = torch.Tensor({5, 5})
covariance = torch.eye(2)

for i = 100,200 do
	local a_sample = distributions.mvn.rnd(mean, covariance)
	x_in_order[1][i] = a_sample[1]
	x_in_order[2][i] = a_sample[2]
        y_in_order[i] = 1
end

x = torch.Tensor(2,200)
y = torch.zeros(200)

for i = 1,200 do
	index = shuffled_index[i]
	
	x[1][i] = x_in_order[1][index]
	x[2][i] = x_in_order[2][index]
	y[i] = y_in_order[index]	
end

--print(x)

params = {
   W = {
      torch.randn(4,2),
      torch.randn(1,4),
   },
   b = {
      torch.randn(4),
      torch.randn(1),
   }
}



neuralNet = function(params, x, y)

   local h1 = torch.tanh(params.W[1]*x  + torch.repeatTensor(params.b[1], 200) )
   local h2 = torch.tanh(params.W[2]*h1 + torch.repeatTensor(params.b[2], 200) )
   
   
   local loss = torch.sum(torch.pow(h2 - y, 2))   

   return loss
end

dneuralNet = autograd(neuralNet)

local max_iteration = 20
local ite = 0


while ite < max_iteration do
	
	local grads, loss = dneuralNet(params, x, y)

	
	for i = 1,#params.W do
      		
		if params.W[i]:dim() == grads.W[i]:dim() then 
      			params.W[i]:add(-.01, grads.W[i])
		else
			print("dimension mismatch")
			return
		end

		
		
                if type(grads.b[i]) == "number" then
			gg =  torch.Tensor({grads.b[i]})
			params.b[i]:add(-.01, gg)

		elseif torch.isTensor(grads.b[i]) then
			params.b[i]:add(-.01, grads.b[i])

		end

		
	end
        
	ite = ite +1
	print("loss = " , loss ,"[",ite,"]")
end

--- test ---

test_x = torch.Tensor(2,200)
test_y = torch.Tensor(200)

mean = torch.Tensor({-5, -5})
covariance = torch.eye(2)

for i = 1,100 do
	local a_sample = distributions.mvn.rnd(mean, covariance)
	test_x[1][i] = a_sample[1]
	test_x[2][i] = a_sample[2]
        test_y[i] = -1
end

mean = torch.Tensor({5, 5})
covariance = torch.eye(2)

for i = 100,200 do
	local a_sample = distributions.mvn.rnd(mean, covariance)
	test_x[1][i] = a_sample[1]
	test_x[2][i] = a_sample[2]
        test_y[i] = 1
end

loss = neuralNet(params,test_x, test_y)
print("\n test loss =", loss)

