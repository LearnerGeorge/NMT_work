#include "niuxor.h"
#include "../../tensor/function/FHeader.h"

namespace niuxor {
	/*base parameter*/
	float learningRate = 0.04F;           // learning rate
	int nEpoch = 1500;                      // max training epochs
	float minmax = 0.5F;                 // range [-p,p] for parameter initialization
	void Init(xorModel &model);
	void InitGrad(xorModel &model, xorModel &grad);
	void Train(float(*trainDataX)[2], float *trainDataY, int dataSize,xorModel &model);
	void Forword(XTensor &input, xorModel &model, xorNet &net);
	void MSELoss(XTensor &output, XTensor &gold, XTensor &loss);
	void Backward(XTensor &input, XTensor &gold, xorModel &model, xorModel &grad, xorNet &net);
	void Update(xorModel &model, xorModel &grad, float learningRate);
	void CleanGrad(xorModel &grad);
	void Test(float(*testData)[2], int testDataSize, xorModel &model);

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
	void InitGrad(xorModel &model, xorModel &grad)
	{
		InitTensor(&grad.weight1, &model.weight1);
		InitTensor(&grad.weight2, &model.weight2);
		InitTensor(&grad.b, &model.b);

		grad.h_size = model.h_size;
		grad.devID = model.devID;
	}

	void Train(float(*trainDataX)[2], float *trainDataY, int dataSize, xorModel &model)
	{
		TensorList inputList;
		TensorList goldList;
		for (int index = 0; index < dataSize; index++)
		{
			XTensor* inputDate = NewTensor2D(1, 2, X_FLOAT, model.devID);
			for (int i = 0; i < 2; i++)
				inputDate->Set2D(trainDataX[index][i]/10, 0, i);
			inputList.Add(inputDate);

			XTensor* goldDate = NewTensor2D(1, 1, X_FLOAT, model.devID);
			goldDate->Set2D(trainDataY[index]/10, 0, 0);
			goldList.Add(goldDate);
		}

		xorNet net;
		xorModel grad;
		InitGrad(model, grad);
		for (int epochIndex = 0; epochIndex < nEpoch; ++epochIndex)
		{
			float totalLoss = 0;
			if ((epochIndex + 1) % 50 == 0)
				learningRate /= 3;
			for (int i = 0; i < inputList.count; ++i)
			{
				XTensor *input = inputList.GetItem(i);
				//input->Dump(stderr);
				XTensor *gold = goldList.GetItem(i);

				Forword(*input, model, net);
				//output.Dump(stderr);
				XTensor loss;
				MSELoss(net.output, *gold, loss);
				totalLoss += loss.Get1D(0);

				//loss.Dump(stderr);
				Backward(*input, *gold, model, grad, net);
				Update(model, grad, learningRate);

				CleanGrad(grad);
			}
			if (epochIndex % 10 == 0) {
				printf("epoch %d\n", epochIndex);
				printf("%f\n", totalLoss / inputList.count);
			}

		}

	}

	void Forword(XTensor &input, xorModel &model, xorNet &net)
	{
		//model.weight1.Dump(stderr);
		net.hidden_state1 = MatrixMul(input, model.weight1);
		//net.hidden_state1.Dump(stderr);
		net.hidden_state2 = net.hidden_state1 + model.b;
		net.hidden_state3 = Sigmoid(net.hidden_state2);
		net.output = MatrixMul(net.hidden_state3, model.weight2);
	}

	void MSELoss(XTensor &output, XTensor &gold, XTensor &loss)
	{
		XTensor tmp = output - gold;
		loss = ReduceSum(tmp, 1, 2) / output.dimSize[1];
	}
	void MSELossBackword(XTensor &output, XTensor &gold, XTensor &grad)
	{
		XTensor tmp = output - gold;
		grad = tmp * 2;
	}

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

	void Update(xorModel &model, xorModel &grad, float learningRate)
	{
		model.weight1 = Sum(model.weight1, grad.weight1, -learningRate);
		model.weight2 = Sum(model.weight2, grad.weight2, -learningRate);
		model.b = Sum(model.b, grad.b, -learningRate);
	}

	void CleanGrad(xorModel &grad)
	{
		grad.b.SetZeroAll();
		grad.weight1.SetZeroAll();
		grad.weight2.SetZeroAll();
	}

	void Test(float(*testData)[2], int testDataSize, xorModel &model)
	{
		xorNet net;
		XTensor*  inputData = NewTensor2D(1, 2, X_FLOAT, model.devID);
		for (int index = 0; index < testDataSize; ++index)
		{
			for (int i = 0; i < 2; i++)
				inputData->Set2D(testData[index][i]/10, 0, i);

			Forword(*inputData, model, net);
			float ans = net.output.Get2D(0, 0);
			printf("%.f ^ %.f = %.f\n", testData[index][0], testData[index][1], ans*10);
		}




	}

}