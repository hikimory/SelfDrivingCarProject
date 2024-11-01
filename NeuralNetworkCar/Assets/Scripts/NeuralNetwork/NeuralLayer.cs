using NeuralNetwork.ActivationFunctions;
using System;

namespace NeuralNetwork
{
    public class NeuralLayer
    {
        public Neuron[] Neurons { get; private set; }

        public NeuralLayer(int neuronCount, int inputCount, IActivationFunction funct)
        {
            Neurons = new Neuron[neuronCount];

            for (int i = 0; i < neuronCount; i++)
            {
                Neurons[i] = new Neuron(inputCount, funct);
            }
            
        }

        public void SetActivationFunction(int index, IActivationFunction funct)
        {
            Neurons[index].SeActivationFunction(funct);
        }

        public double[] Calculate(double[] inputs)
        {
            double[] outputs = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                outputs[i] = Neurons[i].Activate(inputs);
            }
            return outputs;
        }

        public void CalculateDeltas(double[] expected)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].CalculateDelta(expected[i]);
            }
        }

        public void CalculateHiddenDeltas(double[] sums)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].CalculateHiddenDeltas(sums[i]);
            }
        }

        public double[] GetDeltasSums(int neuronCount)
        {
            double[] sums = new double[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                double sum = 0;
                for (int j = 0; j < Neurons.Length; j++)
                {
                    sum += Neurons[j].CalculateHiddenDeltas(i);
                }
                sums[i] = sum;
            }
            
            return sums;
        }

        public double[] GetDeltas()
        {
            double[] deltas = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                deltas[i] = Neurons[i].Delta;
            }
            return deltas;
        }

        public double[] GetValues()
        {
            double[] values = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                values[i] = Neurons[i].Value;
            }
            return values;
        }

        public double[][] GetWeights()
        {
            double[][] weights = new double[Neurons.Length][];
            for (int i = 0; i < Neurons.Length; i++)
            {
                weights[i] = Neurons[i].Weights;
            }
            return weights;
        }

        public void SetWeigths(double[][] weights)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].SetWeights(weights[i]);
            }
        }

        public void SetBiases(double[] biases)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].SetBias(biases[i]);
            }
        }

        public void UpdateWeights(double[] inputs, double learningRate, double gamma)
        {
            foreach (var neuron in Neurons)
            {
                neuron.UpdateWeights(inputs, learningRate, gamma);
            }
        }
    }

}
