using NeuralNetwork.ActivationFunctions;
using NeuralNetwork.Utils;

namespace NeuralNetwork
{
    public class Neuron
    {
        public IActivationFunction ActivationFunction { get; private set; }
        public double[] Weights { get; private set; }
        public double[] PrevWeights { get; private set; }
        public double Bias { get; private set; }
        public double Value { get; private set; }
        public double Delta { get; private set; }

        public Neuron(int inputCount, IActivationFunction activationFunction)
        {
            Weights = new double[inputCount];
            PrevWeights = new double[inputCount];
            ActivationFunction = activationFunction;
            Bias = CryptoRandom.NextDouble() * 2 - 1;
            for (int i = 0; i < inputCount; i++)
            {
                Weights[i] = CryptoRandom.NextDouble() * 2 - 1;
                PrevWeights[i] = 0;
            }
        }

        public double Activate(double[] inputs)
        {
            double sum = 0;
            for (int i = 0; i < Weights.Length; i++)
            {
                sum += Weights[i] * inputs[i];
            }
            sum += Bias;
            Value = ActivationFunction.CalculateOutput(sum);
            return Value;
        }

        public void CalculateDelta(double expected)
        {
            Delta = -(expected - Value) * ActivationFunction.CalculateDerivative(Value);
        }

        public void CalculateHiddenDeltas(double sum)
        {
            Delta = sum * ActivationFunction.CalculateDerivative(Value);
        }

        public double CalculateHiddenDeltas(int index)
        {
            return Weights[index] * Delta;
        }

        public void SetWeights(double[] inputs)
        {
            Weights = inputs;
        }

        public void SetBias(double bias)
        {
            Bias = bias;
        }

        public void SeActivationFunction(IActivationFunction funct)
        {
            ActivationFunction = funct;
        }

        public void UpdateWeights(double[] inputs, double learningRate, double gamma)
        {
            for (int i = 0; i < Weights.Length; i++)
            {
                var deltaWeight = learningRate * Delta * inputs[i] + gamma * PrevWeights[i];
                Weights[i] -= deltaWeight;
                PrevWeights[i] = deltaWeight;
            }
            Bias -= learningRate * Delta;
        }
    }

}
