
### 利用前馈神经网络实现0~7异或操作

利用前馈神经网络实现异或操作<br>

#### step1:前馈网络初始化


```
void Init(xorModel &model)
	{
		InitTensor2D(&model.weight1, 2, model.h_size, X_FLOAT, model.devID);
		InitTensor2D(&model.weight2, model.h_size, 1, X_FLOAT, model.devID);
		InitTensor2D(&model.b, 1, model.h_size, X_FLOAT, model.devID);
		model.weight1.SetDataRand(-minmax, minmax);
		model.weight2.SetDataRand(-minmax, minmax);
		model.b.SetZeroAll();
		printf("Init All!\n");
	}

	
```
#### step2:训练数据
```

int niuxorMain(int argc, const char ** argv)
	{
		xorModel model;
		model.h_size = 20;
		const int dataSize = 64;
		const int dataCol = 2;
		const int dataTestSize = 8;
		model.devID = -1;
		Init(model);
		float  trainDateX[dataSize][dataCol] = { 0,0,0,1,0,2,0,3,0,4,0,5,0,6,0,7,
			1,0,1,1,1,2,1,3,1,4,1,5,1,6,1,7,
			2,0,2,1,2,2,2,3,2,4,2,5,2,6,2,7,
			3,0,3,1,3,2,3,3,3,4,3,5,3,6,3,7,
			4,0,4,1,4,2,4,3,4,4,4,5,4,6,4,7,
			5,0,5,1,5,2,5,3,5,4,5,5,5,6,5,7,
			6,0,6,1,6,2,6,3,6,4,6,5,6,6,6,7,
			7,0,7,1,7,2,7,3,7,4,7,5,7,6,7,7 };
		float testDateX[dataTestSize][dataCol] = { 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7 };

		float trainDateY[dataSize] = { 0 };
		for (int index = 0; index < dataSize; index++)
		{
			trainDateY[index] = int(trainDateX[index][0]) ^ int(trainDateX[index][1]);
		}
		Train(trainDateX, trainDateY, dataSize, model);
		Test(testDateX, dataTestSize, model);
		return 0;

	}

	

```
#### step3 定义前向
```

void Forword(XTensor &input, xorModel &model, xorNet &net)
	{
		//model.weight1.Dump(stderr);
		net.hidden_state1 = MatrixMul(input, model.weight1);
		//net.hidden_state1.Dump(stderr);
		net.hidden_state2 = net.hidden_state1 + model.b;
		net.hidden_state3 = Sigmoid(net.hidden_state2);
		net.output = MatrixMul(net.hidden_state3, model.weight2);
	}


```
#### step4 反向传播计算梯度
```

void Backward(XTensor &input, XTensor &gold, xorModel &model, xorModel &grad, xorNet &net)
	{
		XTensor lossGrad;
		XTensor &dedw2 = grad.weight2;
		XTensor &dedb = grad.b;
		XTensor &dedw1 = grad.weight1;
		MSELossBackword(net.output, gold, lossGrad);
		MatrixMul(net.hidden_state3, X_TRANS, lossGrad, X_NOTRANS, dedw2);
		XTensor dedy = MatrixMul(lossGrad, X_NOTRANS, model.weight2, X_TRANS);
		_SigmoidBackward(&net.hidden_state3, &net.hidden_state2, &dedy, &dedb);
		//dedb.Dump(stderr);
		//input.Dump(stderr);
		dedw1 = MatrixMul(dedb, X_TRANS, input, X_NOTRANS);
	}


```
#### step5 更新模型
```

void Update(xorModel &model, xorModel &grad, float learningRate)
	{
		model.weight1 = Sum(model.weight1, grad.weight1, -learningRate);
		model.weight2 = Sum(model.weight2, grad.weight2, -learningRate);
		model.b = Sum(model.b, grad.b, -learningRate);
	}

```

