using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.ActivationFunctions
{
    public class ThresholdFuncion : IActivationFunction
    {
        public double CalculateDerivative(double input)
        {
            return 0;
        }

        public double CalculateOutput(double input)
        {
            return input >= 0 ? 1 : 0;
        }
    }
}
