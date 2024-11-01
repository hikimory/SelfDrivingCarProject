using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.ActivationFunctions
{
    public class SigmoidActivationFunction : IActivationFunction
    {
        private double _coeficient;

        public SigmoidActivationFunction(double coeficient = 0.5)
        {
            _coeficient = coeficient;
        }

        public double CalculateDerivative(double input)
        {
            return input * (1 - input);
        }

        public double CalculateOutput(double input)
        {
            //return 1.0 / (1.0 + Math.Exp(-input * _coeficient));
            return 1.0 / (1.0 + Math.Exp(-input));
        }
    }
}
