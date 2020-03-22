#include<torch/torch.h>
#include<iostream.h>

struct linearRegression : torch::nn::Module
{
	linearRegression() {
		linear = register_module("linear", torch::nn::Linear(1, 1));
	}

	torch::Tensor forward(torch::Tensor X)
	{
		return linear->forward(X);
	}

	torch::nn::Linear linear{nullptr};
};

int main()
{
	std::cout<<"---------------LINEAR REGRESSION---------------";

	torch::Tensor X = torch::linspace(1, 50, 50).reshape({-1, 1});
	torch::Tensor noise = torch::randint(-8, 9, (50, 1), dtype=torch::float);
	torch Tensor y = 2 * X + 1 + noise;

	auto linear_model = std::make_shared<linearRegression>();

	torch::optim::SGD optimizer(linear_model->parameters(), torch::optim::SGDOptions(0.0001));
	const int epochs = 100;

	for(int epoch=0; epoch!=epochs; epoch++)
	{
		torch::Tensor pred = linear_model->forward(X);
		torch::Tensor loss = torch::nn::functional::mse_loss(pred, y);

		if(((epoch+1)%10 == 0) || (epoch==0) || (epoch==(epochs-1)))
		{
			std::cout<<"Epoch "<< (epoch+1) << "\nLoss: " << loss.item<float>() << std::endl;
		}

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

	}	

	torch::save(net, "models/linear_model.pt");
}