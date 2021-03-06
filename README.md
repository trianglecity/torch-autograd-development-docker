
Torch-autograd development environment is implemented using Docker.

The Makefile is based on https://docs.docker.com/opensource/project/set-up-dev-env.

NOTICE1: The toy (classification) example is based on (theano) http://colinraffel.com/talks/next2015theano.pdf.



Please follow the instructions to run the toy example.

[1] download (or git clone) this source code folder.

[2] cd downloaded-source-code-folder.

[3] sudo make BIND_DIR=.  shell

[4] wait ... wait ... then a bash shell will be ready (root@18f7c7884c72:/#).

[5] root@18f7c7884c72:/# cd /home/ml/lua

[6] root@18f7c7884c72:/home/ml/lua# cd torch

[7] root@18f7c7884c72:/home/ml/lua/torch# ./clean.sh

[8] root@18f7c7884c72:/home/ml/lua/torch# TORCH_LUA_VERSION=LUA53 ./install.sh

[9] type in yes [and enter].

[10] root@18f7c7884c72:/home/ml/lua/torch# source /root/.bashrc

[11] root@18f7c7884c72:/home/ml/lua/torch# cd ..

[12] root@18f7c7884c72:/home/ml/lua# cd torch-autograd

[12] root@18f7c7884c72:/home/ml/lua/torch-autograd# luarocks make

[13] root@18f7c7884c72:/home/ml/lua/torch-autograd# cd ..

[14] root@18f7c7884c72:/home/ml/lua# luarocks install https://raw.github.com/jucor/torch-distributions/master/distributions-0-0.rockspec

[15] root@18f7c7884c72:/home/ml/lua# luarocks install nn

[16] root@18f7c7884c72:/home/ml/lua# cd example/

[17] root@18f7c7884c72:/home/ml/lua/example# lua ./classification_gaussian.lua



The example code (from a theano tutorial) is as follows.

	
	torch = require 'torch'
	autograd = require 'autograd'
	dist = require 'distributions'

	shuffeled_index = torch.randperm(200) 
	-- print(shuffeled_index) -- 1 to 200

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
		index = shuffeled_index[i]
	
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

			print("\n")
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




Customized messages from the FIRST iteration regarding nodes look like



		root@18f7c7884c72:/home/ml/lua/example# lua classification_gaussian.lua
		
		
		... ... ... ... gradOpt main-grad	nil
		... ... ... ... ... func	nparams	3
		... ... ... ... ... func	lastlinedefined	67
		... ... ... ... ... func	linedefined	58
		... ... ... ... ... func	short_src	classification_gaussian.lua
		... ... ... ... ... func	source	@classification_gaussian.lua
		
		
		[1] ... main.lua : opt.optimize is not provided
		
		[2] ... ... runtime.direct init.lua : opt.withForward 	true
		[2] ... ... runtime.direct init.lua : opt.withGradients 	true
		
		    ... ... ... opt.argnum = 	1
		    ... ... ... ... fn is function 	function: 0xc10490
		
		   ... ... ... ... ... .... 	nparams	3
		   ... ... ... ... ... .... 	lastlinedefined	67
		   ... ... ... ... ... .... 	linedefined	58
		   ... ... ... ... ... .... 	short_src	classification_gaussian.lua
		   ... ... ... ... ... .... 	source	@classification_gaussian.lua
		    ... ... ... type(opt) 	table
		   ... ... ... ... 	protected	false
		   ... ... ... ... 	withGradients	true
		   ... ... ... ... 	showCode	false
		   ... ... ... ... 	profileReportFrequency	10
		   ... ... ... ... 	withForward	true
		   ... ... ... ... 	profile	off
		   ... ... ... ... 	reduceFootprint	false
		   ... ... ... ... 	partialGrad	false
		   ... ... ... ... 	argnum	1
		   ... ... ... ... 	optimize	false
		
		[3] ... ... ... DirectTape.lua (runtime) : function grad
		 
		[4-1] ... ... ... DirectTape.lua (runtime) : function funOnly
		[4-2]   ... ... ... ...  type(fun) 	function
		   ... ... ... ...  ... 	nparams	3
		
		   ... ... ... ...  ... 	lastlinedefined	67
		   ... ... ... ...  ... 	linedefined	58
		   ... ... ... ...  ... 	short_src	classification_gaussian.lua
		   ... ... ... ...  ... 	source	@classification_gaussian.lua
		
		[4-3]    ... ... ... ... newStartNode
		      ... ... ... ... ... ... 	W	table: 0xc0fea0
		      ... ... ... ... ... ... 	b	table: 0xc0ff00
		
			[5] ... ... ... ... DirectNode.lua (runtime) : newStartNode
			[5-4]... ... ... ... ... arg[argnum] is table
			[5-5]... ... ... ... ... type(tape) is 	table
		           ... ... ... ... ... ... start iteration in table
		
			[5] ... ... ... ... DirectNode.lua (runtime) : newStartNode
			[5-4]... ... ... ... ... arg[argnum] is table
			[5-6]... ... ... ... ... ... tensor	1
			     ... ... ... ... ... ... ... rows 	4
			     ... ... ... ... ... ... ... columns 	2
			[5-6]... ... ... ... ... ... tensor	2
			     ... ... ... ... ... ... ... rows 	1
			     ... ... ... ... ... ... ... columns 	4
			[5-5]... ... ... ... ... type(tape) is 	table
		           ... ... ... ... ... ... start iteration in table
		
			[5] ... ... ... ... DirectNode.lua (runtime) : newStartNode
			[5-1]... ... ... ... ... arg[argnum] is tensor
			     ... ... ... ... ... ... rows 	4
			     ... ... ... ... ... ... columns 	2
			
			[5-3] ... ... ... ... ... ... ... getting a new node
			[5-7]... ... ... ... DirectNode (runtime): init (new node) 
			[5-7]... ... ... ... ... a new node is ready
		
			[5] ... ... ... ... DirectNode.lua (runtime) : newStartNode
			[5-1]... ... ... ... ... arg[argnum] is tensor
			     ... ... ... ... ... ... rows 	1
			     ... ... ... ... ... ... columns 	4
			[5-2]... ... ... ... ... type(tape)	table
			[5-3] ... ... ... ... ... ... ... getting a new node
			[5-7]... ... ... ... DirectNode (runtime): init (new node) 
			[5-7]... ... ... ... ... a new node is ready
		
		
			[5] ... ... ... ... DirectNode.lua (runtime) : newStartNode
			[5-4]... ... ... ... ... arg[argnum] is table
			[5-6]... ... ... ... ... ... tensor	1
			     ... ... ... ... ... ... ... rows 	4
			[5-6]... ... ... ... ... ... tensor	2
			     ... ... ... ... ... ... ... rows 	1
			[5-5]... ... ... ... ... type(tape) is 	table
		           ... ... ... ... ... ... ... 	1	table: 0xc11810
		           ... ... ... ... ... ... ... 	2	table: 0xc11e70
		
		           ... ... ... ... ... ... start iteration in table
		
			[5] ... ... ... ... DirectNode.lua (runtime) : newStartNode
			[5-1]... ... ... ... ... arg[argnum] is tensor
			     ... ... ... ... ... ... rows 	4
			[5-2]... ... ... ... ... type(tape)	table
			[5-2]... ... ... ... ... ... n_elements in tape (table) 	3
			[5-3] ... ... ... ... ... ... ... getting a new node
			[5-7]... ... ... ... DirectNode (runtime): init (new node) 
			[5-7]... ... ... ... ... a new node is ready
		
		
			[5] ... ... ... ... DirectNode.lua (runtime) : newStartNode
			[5-1]... ... ... ... ... arg[argnum] is tensor
			     ... ... ... ... ... ... rows 	1
			[5-2]... ... ... ... ... type(tape)	table
			[5-2]... ... ... ... ... ... n_elements in tape (table) 	4
			[5-3] ... ... ... ... ... ... ... getting a new node
			[5-7]... ... ... ... DirectNode (runtime): init (new node) 
			[5-7]... ... ... ... ... a new node is ready
		
		[4-4]    ... ... ... ... type(arg[argnum]) 	table
		            ... ... ... ... ... ... 	W	table: 0xc11750
		            ... ... ... ... ... ... 	b	table: 0xc12230
		[4-5]    ... ... ... ... overload.install
		
		     [7]... ... ... ... overload.lua : install(nodeApply)
		     [7]... ... ... ... ... type(fn) 	function
		 
		            ... ... 	lastlinedefined	283
		            ... ... 	linedefined	19
		            ... ... 	short_src	...all/share/lua/5.3/autograd/runtime/direct/DirectTape.lua
		            ... ... 	source	@/home/ml/lua/torch/install/share/lua/5.3/autograd/runtime/direct/DirectTape.lua
		
		    ... ... ... ... ... ... ..  #toRegister 	4
		
		    ... ... ... ... ... ... ..  toRegister 	function: 0xbc1690
		
		     [7]  ... ... in overload module
		      ... ... ... name 	torch
		     [7]  ... ... ... ... ... fn(moduleFns)
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	unm	85
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	add	40
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	div	79
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	mul	80
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	sub	92
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	pow	20
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	unm	34
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	add	77
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	div	28
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	mul	56
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	sub	48
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	pow	63
		
		    ... ... ... ... ... ... ..  toRegister 	function: 0xbc17c0
		
		     [7]  ... ... in overload module
		      ... ... ... name 	Value
		     [7]  ... ... ... ... ... fn(moduleFns)
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	unm	37
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	add	52
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	div	96
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	mul	92
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	sub	64
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	pow	72
		
		    ... ... ... ... ... ... ..  toRegister 	function: 0xbc1920
		
		     [7]  ... ... in overload module
		      ... ... ... name 	DirectNode
		     [7]  ... ... ... ... ... fn(moduleFns)
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	unm	15
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	add	61
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	div	2
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	mul	25
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	sub	14
		   ... ... ... ... ... ... ... ... ... (overload module ) overloadOp	pow	81
		
		    ... ... ... ... ... ... ..  toRegister 	function: 0xbc1aa0
		
		     [7]  ... ... in overload module
		      ... ... ... name 	util
		     [7]  ... ... ... ... ... fn(moduleFns)
		    ... ... ... ... ... ... ... 
		... ... ... ... nnwrapper.lua : setApply(fn)
		... ... ... ... ... type(fn) 	function
		      ... ... ... 	lastlinedefined	283
		      ... ... ... 	linedefined	19
		      ... ... ... 	short_src	...all/share/lua/5.3/autograd/runtime/direct/DirectTape.lua
		      ... ... ... 	source	@/home/ml/lua/torch/install/share/lua/5.3/autograd/runtime/direct/DirectTape.lua
		
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.getmetatable	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.getmetatable	0
		
		[4-6]   ... ... ... ... neuralNet = function(params, x, y)
		[4-6]    ... ... ... ... arg 	table: 0xc11470
		       ... ... ... ... ... 	1	table
		       ... ... ... ... ... ... ... 	W	table: 0xc11750
		       ... ... ... ... ... ... ... 	b	table: 0xc12230
		       ... ... ... ... ... 	2	userdata
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.size	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.size	0
		       ... ... ... ... ... ... ... x 	nil	2	 by 	200
		       ... ... ... ... ... 	3	userdata
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.size	0
		       ... ... ... ... ... ... ... y 	nil	200
		
		
		[4-7]   ... ... ... ... ... neuralNet(table.unpack(arg))
		[4-7]   ... ... ... ... ... fun(table.unpack(arg)) 
		
		   ... ... ... ... ... ... ... (overlod module ) overloadOp nodeApply, newFn 	function	__mul	25
		  [6] ... ... ... ... ... ...  nodeApply	op.__mul	0
		  [6] ... ... ... ... ... DirectTape (runtime) : nodeApply
		  [6] ... ... ... ... ... ... fun.name 	op.__mul
		   
		   ... ... ... ... ... ... ...  __mul returns	userdata	*	userdata
		   ... ... ... ... ... ... ...  (overlod module ) overloadOp nodeApply, newFn 	function	__mul	56
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	op.__mul	0
		    ... ... ... ... ... ... ... gettin a new node
		
		    ... ... ... ... ... ... ... type(tape) 	table
		    ... ... ... ... ... ... ...  	fn	function: 0xb5bd50
		    ... ... ... ... ... ... ...  	capture	true
		    ... ... ... ... ... ... ...  	operator	mul
		    ... ... ... ... ... ... ...  	differentiable	true
		    ... ... ... ... ... ... ...  	name	op.__mul
			[5-7]... ... ... ... DirectNode (runtime): init (new node) 
			[5-7]... ... ... ... ... a new node is ready
		
		   
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.isTensor	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.repeatTensor	0
		  [6] ... ... ... ... ... DirectTape (runtime) : nodeApply
		  [6] ... ... ... ... ... ... fun.name 	torch.repeatTensor
		   
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.type	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.new	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.isContiguous	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.type	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.new	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.set	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.size	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.cmul	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.long	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.type	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.getmetatable	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.nElement	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.size	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.resize	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.resize	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.new	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.size	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.size	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.unfold	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.resize	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.expandAs	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.size	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.expand	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.type	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.new	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.type	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.stride	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.size	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.storage	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.storageOffset	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.set	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.copy	0
		    ... ... ... ... ... ... ... gettin a new node
		
		    ... ... ... ... ... ... ... type(tape) 	table
		    ... ... ... ... ... ... ...  	unsupported	false
		    ... ... ... ... ... ... ...  	capture	true
		    ... ... ... ... ... ... ...  	differentiable	true
		    ... ... ... ... ... ... ...  	fn	function: 0xa979f0
		    ... ... ... ... ... ... ...  	name	torch.repeatTensor
			[5-7]... ... ... ... DirectNode (runtime): init (new node) 
			[5-7]... ... ... ... ... a new node is ready
		
		  
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.isTensor	0
		   ... ... ... ... ... ... ... ... ... (overlod module ) overloadOp nodeApply, newFn 	function	__add	61
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	op.__add	0
		  [6] ... ... ... ... ... DirectTape (runtime) : nodeApply
		  [6] ... ... ... ... ... ... fun.name 	op.__add
		   ... ... ... ... ... ... ... ... table size 	12
		   ... ... ... ... ... ... ... ... __add returns	userdata	+	userdata
		   ... ... ... ... ... ... ... ... ... (overlod module ) overloadOp nodeApply, newFn 	function	__add	77
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	op.__add	0
		    ... ... ... ... ... ... ... gettin a new node
		
		    ... ... ... ... ... ... ... type(tape) 	table
		    ... ... ... ... ... ... ...  	fn	function: 0xb60780
		    ... ... ... ... ... ... ...  	capture	true
		    ... ... ... ... ... ... ...  	operator	add
		    ... ... ... ... ... ... ...  	differentiable	true
		    ... ... ... ... ... ... ...  	name	op.__add
			[5-7]... ... ... ... DirectNode (runtime): init (new node) 
			[5-7]... ... ... ... ... a new node is ready
		
		   ... ... ... ... ... ... ... ... ... (overlod module ) execute nodeApply, fn 	function	isTensor
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.isTensor	0
		   ... ... ... ... ... ... ... ... ... (overlod module ) execute nodeApply, fn 	function	tanh
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.tanh	0
		  [6] ... ... ... ... ... DirectTape (runtime) : nodeApply
		  [6] ... ... ... ... ... ... fun.name 	torch.tanh
		       ... ... ... ... ... ... ... gettin a new node
		
		    ... ... ... ... ... ... ... type(tape) 	table
		    ... ... ... ... ... ... ...  	unsupported	false
		    ... ... ... ... ... ... ...  	capture	true
		    ... ... ... ... ... ... ...  	differentiable	true
		    ... ... ... ... ... ... ...  	fn	function: 0x7ffbcb18cd00
		    ... ... ... ... ... ... ...  	name	torch.tanh
			[5-7]... ... ... ... DirectNode (runtime): init (new node) 
			[5-7]... ... ... ... ... a new node is ready
		
		   
		  [6] ... ... ... ... ... ...  nodeApply	torch.isTensor	0
		   ... ... ... ... ... ... .. (overlod module ) overloadOp nodeApply, newFn 	function	__mul	25
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	op.__mul	0
		  [6] ... ... ... ... ... DirectTape (runtime) : nodeApply
		  [6] ... ... ... ... ... ... fun.name 	op.__mul
		   ... ... ... ... ... ... ... ... table size 	12
		   ... ... ... ... ... ... ... ... __mul returns	userdata	*	userdata
		   ... ... ... ... ... ... ... (overlod module ) overloadOp nodeApply, newFn 	function	__mul	56
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	op.__mul	0
		    ... ... ... ... ... ... ... gettin a new node
		
		    ... ... ... ... ... ... ... type(tape) 	table
		    ... ... ... ... ... ... ...  	fn	function: 0xb5bd50
		    ... ... ... ... ... ... ...  	capture	true
		    ... ... ... ... ... ... ...  	operator	mul
		    ... ... ... ... ... ... ...  	differentiable	true
		    ... ... ... ... ... ... ...  	name	op.__mul
			[5-7]... ... ... ... DirectNode (runtime): init (new node) 
			[5-7]... ... ... ... ... a new node is ready
		
		   
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.isTensor	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.repeatTensor	0
		  [6] ... ... ... ... ... DirectTape (runtime) : nodeApply
		  [6] ... ... ... ... ... ... fun.name 	torch.repeatTensor
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.type	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.new	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.isContiguous	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.type	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.new	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.set	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.size	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.cmul	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.long	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.type	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.getmetatable	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.nElement	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.size	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.resize	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.resize	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.new	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.size	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.size	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.unfold	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.resize	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.expandAs	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.size	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.expand	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.type	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.new	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.type	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.stride	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.size	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.dim	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.storage	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.storageOffset	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.set	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.DoubleTensor.copy	0
		    ... ... ... ... ... ... ... gettin a new node
		
		    ... ... ... ... ... ... ... type(tape) 	table
		    ... ... ... ... ... ... ...  	unsupported	false
		    ... ... ... ... ... ... ...  	capture	true
		    ... ... ... ... ... ... ...  	differentiable	true
		    ... ... ... ... ... ... ...  	fn	function: 0xa979f0
		    ... ... ... ... ... ... ...  	name	torch.repeatTensor
			[5-7]... ... ... ... DirectNode (runtime): init (new node) 
			[5-7]... ... ... ... ... a new node is ready
		
		   
		  [6] ... ... ... ... ... ...  nodeApply	torch.isTensor	0
		   ... ... ... ... ... ... ... (overlod module ) overloadOp nodeApply, newFn 	function	__add	61
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	op.__add	0
		  [6] ... ... ... ... ... DirectTape (runtime) : nodeApply
		  [6] ... ... ... ... ... ... fun.name 	op.__add
		  
		   ... ... ... ... ... ... ... ... __add returns	userdata	+	userdata
		   ... ... ... ... ... ... ... (overlod module ) overloadOp nodeApply, newFn 	function	__add	77
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	op.__add	0
		    ... ... ... ... ... ... ... gettin a new node
		
		    ... ... ... ... ... ... ... type(tape) 	table
		    ... ... ... ... ... ... ...  	fn	function: 0xb60780
		    ... ... ... ... ... ... ...  	capture	true
		    ... ... ... ... ... ... ...  	operator	add
		    ... ... ... ... ... ... ...  	differentiable	true
		    ... ... ... ... ... ... ...  	name	op.__add
			[5-7]... ... ... ... DirectNode (runtime): init (new node) 
			[5-7]... ... ... ... ... a new node is ready
		
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.isTensor	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.tanh	0
		  [6] ... ... ... ... ... DirectTape (runtime) : nodeApply
		  [6] ... ... ... ... ... ... fun.name 	torch.tanh
		      ... ... ... ... ... ... ... gettin a new node
		
		    ... ... ... ... ... ... ... type(tape) 	table
		    ... ... ... ... ... ... ...  	unsupported	false
		    ... ... ... ... ... ... ...  	capture	true
		    ... ... ... ... ... ... ...  	differentiable	true
		    ... ... ... ... ... ... ...  	fn	function: 0x7ffbcb18cd00
		    ... ... ... ... ... ... ...  	name	torch.tanh
			[5-7]... ... ... ... DirectNode (runtime): init (new node) 
			[5-7]... ... ... ... ... a new node is ready
		
		  [6] ... ... ... ... ... ...  nodeApply	torch.isTensor	0
		   ... ... ... ... ... ... ... (overlod module ) overloadOp nodeApply, newFn 	function	__sub	14
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	op.__sub	0
		  [6] ... ... ... ... ... DirectTape (runtime) : nodeApply
		  [6] ... ... ... ... ... ... fun.name 	op.__sub
		   ... ... ... ... ... ... ... ... __sub returns	userdata	-	userdata
		   ... ... ... ... ... ... .. (overlod module ) overloadOp nodeApply, newFn 	function	__sub	48
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	op.__sub	0
		    ... ... ... ... ... ... ... gettin a new node
		
		    ... ... ... ... ... ... ... type(tape) 	table
		    ... ... ... ... ... ... ...  	fn	function: 0xb5d810
		    ... ... ... ... ... ... ...  	capture	true
		    ... ... ... ... ... ... ...  	operator	sub
		    ... ... ... ... ... ... ...  	differentiable	true
		    ... ... ... ... ... ... ...  	name	op.__sub
			[5-7]... ... ... ... DirectNode (runtime): init (new node) 
			[5-7]... ... ... ... ... a new node is ready
		
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.isTensor	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.pow	0
		  [6] ... ... ... ... ... DirectTape (runtime) : nodeApply
		  [6] ... ... ... ... ... ... fun.name 	torch.pow
		   
		    ... ... ... ... ... ... ... gettin a new node
		
		    ... ... ... ... ... ... ... type(tape) 	table
		    ... ... ... ... ... ... ...  	unsupported	false
		    ... ... ... ... ... ... ...  	capture	true
		    ... ... ... ... ... ... ...  	differentiable	true
		    ... ... ... ... ... ... ...  	fn	function: 0x7ffbcb193b00
		    ... ... ... ... ... ... ...  	name	torch.pow
			[5-7]... ... ... ... DirectNode (runtime): init (new node) 
			[5-7]... ... ... ... ... a new node is ready
		
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.isTensor	0
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.sum	0
		  [6] ... ... ... ... ... DirectTape (runtime) : nodeApply
		  [6] ... ... ... ... ... ... fun.name 	torch.sum
		      ... ... ... ... ... ... ... gettin a new node
		
		    ... ... ... ... ... ... ... type(tape) 	table
		    ... ... ... ... ... ... ...  	unsupported	false
		    ... ... ... ... ... ... ...  	capture	true
		    ... ... ... ... ... ... ...  	differentiable	true
		    ... ... ... ... ... ... ...  	fn	function: 0x7ffbcb18a3c0
		    ... ... ... ... ... ... ...  	name	torch.sum
			[5-7]... ... ... ... DirectNode (runtime): init (new node) 
			[5-7]... ... ... ... ... a new node is ready
		
		  [6] ... ... ... ... ... ... ... ... ... nodeApply	torch.isTensor	0
		
		[4-8]   ... ... ... ... ... type(allAns) 	table
		... ... ... ... nnwrapper.lua : setApply(fn)
		... ... ... ... ... type(fn) 	nil
		
		... ... type(ans) is a table
		... ... ... ... ... n_elements in ans 	12
		... ... ... ... ... ... ... ... table	args
		... ... ... ... ... ... ... ... elements in table 	1
		... ... ... ... ... ... ... ... ... table
		... ... ... ... ... ... ... ... table	gradFun
		... ... ... ... ... ... ... ... elements in table 	1
		... ... ... ... ... ... ... ... ... function 
		... ... ... ... ... ... ... ... table	argValues
		... ... ... ... ... ... ... ... elements in table 	1
		... ... ... ... ... ... ... ... ... Tensor
		... ... ... ... ... ... ... ... table	fun
		... ... ... ... ... ... ... ... elements in table 	5
		... ... ... ... ... ... ... ... ... function 
		... ... ... ... ... ... ... ... ... string 	nil
		   ... ... ... ... ... isNode(n) 	table: 0x10de660
		... ... ... ... ... DirectTape.gradOnly
